import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns

from argparse import ArgumentParser
from enum import Enum
from lightgbm import LGBMRegressor
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sktime.forecasting.compose import RecursiveReductionForecaster
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.var import VAR
from sktime.performance_metrics.forecasting import MeanAbsoluteError
from sktime.split import ExpandingWindowSplitter


class ModelType(Enum):
    """Enum for type-safe model type."""
    VAR = "VAR"
    LGBM = "LGBM"


def load_data(data_path: str) -> pd.DataFrame:
    """Load the data from a CSV file, set parsed time column as index."""
    data = pd.read_csv(data_path, sep=";", parse_dates=["Time"], index_col="Time")
    data.index = pd.to_datetime(data.index, format='ISO8601', utc=True)
    return data


def scale(scaler: MinMaxScaler, data: pd.DataFrame) -> pd.DataFrame:
    """Wrapper to transform data with a fitted scaler, returning pd.DataFrame."""
    data_scaled_raw = scaler.transform(data)
    data_scaled = pd.DataFrame(
        data_scaled_raw, 
        index=data.index, 
        columns=data.columns
    )
    return data_scaled


def inverse_scale(scaler: MinMaxScaler, data: pd.DataFrame) -> pd.DataFrame:
    """Wrapper to inverse data with a fitted scaler, returning pd.DataFrame."""
    data_inverse_scaled_raw = scaler.inverse_transform(data)
    data_inverse_scaled = pd.DataFrame(
        data_inverse_scaled_raw,
        index=data.index,
        columns=data.columns
    )
    return data_inverse_scaled


def drop_constant_cols(data: pd.DataFrame, target_quantity: str) -> pd.DataFrame:
    """Drop all constant columns from the data.
    
    If the target quantity is constant, raise an error and inform the user.
    """
    constant_cols = [col for col in data.columns if data[col].nunique() <= 1]
    if constant_cols:
        # If the target quantity is constant, raise an error and inform the user
        # that the model would not make sense
        if target_quantity is not None and target_quantity in constant_cols:
            val = data[target_quantity].unique()[0]
            raise ValueError(f"'{target_quantity}' is constant ({val}) and doesn't need regression.")
        data = data.drop(columns=constant_cols)
    return data


def get_endo_exo(data: pd.DataFrame, quantity: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data into endogenous and exogenous series.
    
    The endogenous series is just the quantity, while the exogenous series is the rest.
    """
    endo = data.drop(columns=[col for col in data.columns if col != quantity])
    exo = data.drop(columns=[quantity])
    return endo, exo


def preprocess_and_split(
    data: pd.DataFrame, 
    train_portion: float,
    target_quantity: str,
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """Preprocess and split the data into training and testing subsets.

    Removes constant columns, interpolates missing values, splits and scales
    the data. Returns the training and testing data, as well as a scaler for 
    inverse transformation.
    """
    # Interpolate missing values 
    data = data.interpolate()
    data = data.asfreq("30min")

    # Remove constant columns that may cause fitting issues and are not useful
    data = drop_constant_cols(data, target_quantity)

    # Split the data into training and testing sets so that we can process the
    # data separately and avoid data leakage
    train_size = int(len(data) * train_portion)
    train, test = data.iloc[:train_size], data.iloc[train_size:]

    # Simple zero-one scaling to scale the features to the same range
    scaler = MinMaxScaler()
    scaler = scaler.fit(train)
    train_scaled = scale(scaler, train)
    test_scaled = scale(scaler, test)
    return train_scaled, test_scaled, scaler


def lineplot_series(data: pd.DataFrame, quantity: str, forecast=False):
    """Wrapper to lineplot the time series data, not setting any plot params."""
    if forecast:
        label = f"Forecast {quantity}"
        sns.lineplot(x='Time', y=quantity, data=data, label=label) 
    else:
        label = f"Actual {quantity}"
        sns.lineplot(x='Time', y=quantity, data=data, label=label, linestyle="dashed") 


class TimeSeriesModel:
    """A wrapper for a time-series model. Model can be either VAR or LGBM.

    VAR is a vector autoregression model, very simple statistical model
    that forecasts vector of all quantities as output based on purely past values.

    LGBM is a gradient boosting model that is adapted for forecasting using a
    rolling window approach. We use it to predict the target quantity value using 
    its past values and values of other quantities as features.

    Attributes:
        quantity: Name of the target quantity to be predicted
        model_type: Type of model to be used (VAR or LGBM)
        verbose: If True, print additional information during processing
    Later, raw_data, train, test, and forecast_test attributes are set.
    """
    def __init__(self, quantity: str, model_type: ModelType, verbose: bool = False):
        self.quantity = quantity
        self.model_type = model_type
        self.verbose = verbose

    def fit_and_predict(
        self, 
        raw_data: pd.DataFrame, 
        train_portion: float = 0.8,
    ) -> "TimeSeriesModel":
        """
        Split series into train and test parts, fit model to the train data,
        and forecast for the test part. The split is done so that the training
        data is the first portion of the series and the test data is the last part.

        All the datasets are saved as attributes of this instance for later use.
        """
        self.raw_data = raw_data
        train, test, scaler = preprocess_and_split(self.raw_data, train_portion, self.quantity)
        features = train.columns

        # Choose the lag (window size) parameter for the model using cross-val scores
        lag = self.find_best_lag(train, lag_options=[12, 24, 48])

        if self.model_type == ModelType.LGBM:
            # Split the data into endogenous and exogenous variables
            train_endo, train_exo = get_endo_exo(train, self.quantity)
            test_endo, test_exo = get_endo_exo(test, self.quantity)

            # Fit the LGBM model using windowed data
            regressor = LGBMRegressor(verbose=-1)
            forecaster = RecursiveReductionForecaster(regressor, window_length=lag)
            forecaster = forecaster.fit(y=train_endo, X=train_exo)
            # Forecast for the test set
            forecast_horizon = np.arange(1, len(test_endo) + 1)
            forecast_test = forecaster.predict(fh=forecast_horizon, X=test_exo)
        else:
            # Fit the VAR model on the training data
            forecaster = VAR(maxlags=lag, ic='aic')
            forecaster = forecaster.fit(train)
            # Forecast for the test set
            forecast_horizon = np.arange(1, len(test) + 1)
            forecast_test = forecaster.predict(fh=forecast_horizon)
        forecast_test = pd.DataFrame(forecast_test, index=test.index, columns=features)

        self.train = train
        self.test = test
        self.forecast_test = forecast_test
        self.scaler = scaler
        return self
    
    def find_best_lag(self, data: pd.DataFrame, lag_options: list[int]) -> int:
        """Choose one of the specified lag options by cross-validation scores."""
        cv_mae_results = {}
        for lag in lag_options:
            cv_mae_results[lag] = self.run_cross_val(lag, data)

        # Choose the lag with the lowest mean MAE
        best_lag = min(cv_mae_results, key=cv_mae_results.get)
        if self.verbose:
            score = cv_mae_results[best_lag]
            print(f"Best lag value: {best_lag} (with mean cross-val MAE: {score})\n")
        return best_lag
    
    def run_cross_val(self, lag: int, data: pd.DataFrame) -> float:
        """Run cross-val with the model and specified lag and compute mean MAE 
        score across the splits."""
        # Start with a 10-day initial window, move by 1 day, forecast 1 day ahead
        cv = ExpandingWindowSplitter(initial_window=480, step_length=48, fh=np.arange(1, 48))
        loss = MeanAbsoluteError()

        if self.model_type == ModelType.LGBM:
            regressor = LGBMRegressor(verbose=-1)
            forecaster = RecursiveReductionForecaster(regressor, window_length=12) # TODO
            data_endo, data_exo = get_endo_exo(data, self.quantity)
            results = evaluate(forecaster, y=data_endo, X=data_exo, cv=cv, scoring=loss)
        else:
            forecaster = VAR(maxlags=lag, ic='aic')
            results = evaluate(forecaster, y=data, cv=cv, scoring=loss)

        if self.verbose:
            print(f"Cross-validation results for lag {lag}:\n{results}\n")
        return results['test_MeanAbsoluteError'].mean()
    
    def evaluate_and_plot(self, show_train=False, output_path=None) -> tuple[float, float]:
        """Evaluate the forecasting using MAE score and plot the series.
        
        Mean absolute error (MAE) is used, since it is an intuitive measure. 
        The MAE is computed on both the normalized values (to be comparable 
        across different quantities), as well as at the original scale.

        Plots the actual testing time series against the forecasted values.
        If `show_train` is True, also plots the initial part of the time series
        that was used as the training data.
        """
        mae_instance = MeanAbsoluteError() 
        mae_norm = mae_instance(self.test[self.quantity], self.forecast_test[self.quantity])  

        # Inverse scale the zero-one normalized values to get the original scale
        self.forecast_test = inverse_scale(self.scaler, self.forecast_test)
        self.test = inverse_scale(self.scaler, self.test)
        mae_orig = mae_instance(self.test[self.quantity], self.forecast_test[self.quantity])
        
        plt.figure(figsize=(11, 6))
        if show_train:
            # Plot whole series and highlight different periods in the plot
            lineplot_series(self.raw_data, self.quantity, forecast=False)
            lineplot_series(self.forecast_test, self.quantity, forecast=True)
            start_train, end_train = self.train.index[0], self.train.index[-1]
            start_test, end_test = self.test.index[0], self.test.index[-1]
            plt.axvspan(start_train, end_train, color='lightblue', alpha=0.2, label="Train period")
            plt.axvspan(start_test, end_test, color='orange', alpha=0.15, label="Forecast period")
        else:
            # Plot only the test data
            lineplot_series(self.test, self.quantity, forecast=False)
            lineplot_series(self.forecast_test, self.quantity, forecast=True)
        plt.xticks(rotation=45)
        plt.tight_layout(pad=2.0)
        plt.legend()
        plt.title(f"{self.model_type.value} model forecast")

        # Either save into file or just show the plot
        if output_path is not None:
            plt.savefig(output_path)
        else:
            plt.show()
        return mae_norm, mae_orig


if __name__ == '__main__':
    # Initialize the environment (set values to avoid warnings)
    os.environ["LOKY_MAX_CPU_COUNT"] = str(2)
    warnings.filterwarnings("ignore", category=UserWarning)
    sns.set_theme(style="whitegrid")

    # Parse command line arguments
    parser = ArgumentParser(
        prog='time_series.py',
        description='Loads csv data, fits a forecaster, returns a plot and a precision metric.'
    )
    parser.add_argument('-i', '--input', required=True, help='Path to the input CSV file')
    parser.add_argument('-q', '--quantity', required=True, help='Quantity to model and plot')
    parser.add_argument('-o', '--output', help='Optional path to export the PNG plot')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose messages')
    parser.add_argument('-m', '--model', help='Model type (VAR or LGBM)', default='LGBM', choices=['VAR', 'LGBM'])
    args = parser.parse_args()

    # Load the data and check that the requested quantity is present
    input_data = load_data(args.input)
    if args.quantity not in input_data.columns:
        print(f"Error: Quantity '{args.quantity}' not found in the data.")
        exit(1)

    # Instatiate the model, do all the processing, and plot the results
    model = TimeSeriesModel(args.quantity, ModelType(args.model), args.verbose)
    try:
        model = model.fit_and_predict(input_data, train_portion=0.8)
        mae_norm, mae_orig = model.evaluate_and_plot(show_train=True, output_path=args.output)
        print(f"MAE for normalized '{args.quantity}' forecast:", mae_norm)
        print(f"MAE for '{args.quantity}' forecast:", mae_orig)
    except ValueError as e:
        print(f"Processing finished with an issue: {e}")
        exit(1)
