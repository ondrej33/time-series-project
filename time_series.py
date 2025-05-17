import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from sklearn.preprocessing import MinMaxScaler
from typing import Optional
from sktime.forecasting.var import VAR
from sktime.forecasting.model_evaluation import evaluate
from sktime.split import ExpandingWindowSplitter
from sktime.performance_metrics.forecasting import MeanAbsoluteError


def load_data(data_path: str) -> pd.DataFrame:
    """Load the data from a CSV file, parse the time column, and set it as the index."""
    data = pd.read_csv(data_path, sep=";", parse_dates=["Time"], index_col="Time")
    data.index = pd.to_datetime(data.index, format='ISO8601', utc=True)
    return data


def preprocess_and_split(
    data: pd.DataFrame, 
    train_portion: float,
    target_quantity: str,
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """Preprocess and split the data into training and testing subsets.

    Removes constant columns, interpolates missing values, splits and scales
    the data. Returns the training and testing data, as well as a scaler for 
    inverse transformation.
    
    # TODO: This is just a placeholder to prototype the functionality.
    """
    # Interpolate missing values 
    data = data.interpolate()
    data = data.asfreq("30min")

    # Remove constant columns that may cause issues with the model fitting and are not useful
    constant_cols = [col for col in data.columns if data[col].nunique() <= 1]
    if constant_cols:
        # If the target quantity is constant, raise an error and inform the user
        # that the model would not ma
        if target_quantity is not None and target_quantity in constant_cols:
            val = data[target_quantity].unique()[0]
            raise ValueError(f"Target quantity '{target_quantity}' is constant (value is {val}) and doesn't need regression.")
        data = data.drop(columns=constant_cols)

    # Split the data into training and testing sets so that we can process the
    # data separately and avoid data leakage
    train_size = int(len(data) * train_portion)
    train, test = data.iloc[:train_size], data.iloc[train_size:]

    # Simple zero-one scaling to scale the features to the same range
    scaler = MinMaxScaler()
    scaler = scaler.fit(train)
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    train_scaled = pd.DataFrame(train_scaled, index=train.index, columns=train.columns)
    test_scaled = pd.DataFrame(test_scaled, index=test.index, columns=test.columns)
    return train_scaled, test_scaled, scaler


class RegressionModel:
    """A wrapper for the regression model.
    
    TODO: This is just a placeholder to prototype the class.
    """
    def __init__(self, quantity: str):
        # TODO: Decide what should be saved in the class (model, data, etc.)
        self.quantity = quantity

    def fit_and_predict(
        self, 
        raw_data: pd.DataFrame, 
        train_portion: float = 0.8,
        print_details: bool = False,
    ) -> "RegressionModel":
        """
        Split series into train and test parts, fit model to the train data,
        and forecast for the test part. The split is done so that the training
        data is the first part of the series and the test data is the last part.

        If `print_details` is True, print progress messages (such as cross val 
        scores) during the process. This controls the verbosity of the output.
        
        TODO: This is just a placeholder to prototype the functionality.
        """
        self.raw_data = raw_data
        train, test, scaler = preprocess_and_split(self.raw_data, train_portion, self.quantity)
        features = train.columns

        # Choose parameters for the VAR model using cross-val scores
        lag = self.find_best_lag(train, lag_options=[24, 48], print_details=print_details)

        # Fit model on the full training set
        model = VAR(maxlags=lag, ic='aic')
        model_fitted = model.fit(train)

        # Forecast for the test set
        forecast_test_steps = len(test)
        forecast_test = model_fitted.predict(fh=np.arange(1, forecast_test_steps))
        forecast_test = pd.DataFrame(forecast_test, index=test.index, columns=features)

        # Save the data for later use
        self.train = train
        self.test = test
        self.scaler = scaler
        self.forecast_test = forecast_test
        return self
    
    def find_best_lag(
        self, 
        data: pd.DataFrame, 
        lag_options: list[int], 
        print_details: bool = False
    ) -> VAR:
        """Choose one of the specified lag options by cross-validation scores.

        Return the VAR model with the best lag option, to be fitted on the whole
        training set.
        """
        cv_mae_results = {}
        for lag in lag_options:
            cv_mae_results[lag] = self.var_cross_val(lag, data, print_details=print_details)

        # Choose the lag with the lowest MAE
        best_lag = min(cv_mae_results, key=cv_mae_results.get)
        if print_details:
            print(f"Best lag value: {best_lag} (with cross-val MAE: {cv_mae_results[best_lag]})\n")
        return best_lag
    
    def var_cross_val(
        self, 
        lag: int, 
        data: pd.DataFrame, 
        print_details: bool = False,
    ) -> VAR:
        """Run cross-val with the VAR model with specified lag.
         
        Returns the mean MAE score.
        """
        model = VAR(maxlags=lag, ic='aic')
        # Start with a 10-day initial window, move by 1 day, forecast 1 day ahead
        cv = ExpandingWindowSplitter(initial_window=480, step_length=48, fh=np.arange(1, 48))
        loss = MeanAbsoluteError()
        results = evaluate(forecaster=model, y=data, cv=cv, scoring=loss)
        if print_details:
            print(f"Cross-validation results for lag {lag}:\n{results}\n")
        return results['test_MeanAbsoluteError'].mean()
    
    def plot(self, show_train: bool = False, output_path: Optional[str] = None):
        """Plot the results of the model.
        
        Plots the actual testing time series against the forecasted values.
        If `show_train` is True, also plots the initial part of the time series
        that was used as the training data.
        """
        # TODO: Properly plot actual vs predicted values

        # Inverse transform the scaled forecast data to original scale for plotting
        forecast_test_scaled = pd.DataFrame(
            self.scaler.inverse_transform(self.forecast_test),
            index=self.forecast_test.index,
            columns=self.forecast_test.columns
        )

        # Prepare the data for plotting
        plt.figure(figsize=(12, 6))
        # If show_train is True, plot the training data as well
        start_idx = 0 if show_train else len(self.train)        
        plt.plot(self.raw_data.index[start_idx:], self.raw_data[self.quantity][start_idx:], label=f"Actual {self.quantity}", linestyle="dashed")
        # Plot the forecasted values in different color
        plt.plot(forecast_test_scaled.index, forecast_test_scaled[self.quantity], label=f"Forecast {self.quantity}")
        if show_train:
            # Highligh different periods in the plot
            plt.axvspan(self.raw_data.index[0], self.raw_data.index[len(self.train)-1], color='lightblue', alpha=0.15, label="Train period")
            plt.axvspan(self.raw_data.index[len(self.train)], self.raw_data.index[-1], color='orange', alpha=0.1, label="Forecast period")
        plt.legend()
        plt.title("Normalized VAR model forecast")

        # Either save into file or just show the plot
        if output_path is not None:
            plt.savefig(output_path)
        else:
            plt.show()

    def evaluate(self) -> float:
        """Evaluate the model using a suitable metric. Print the metric and return it."""
        # TODO: Choose a suitable metric for the time series and selected model 
        mae = np.abs(self.forecast_test - self.test).mean()
        print(f"Normalized MAE for {self.quantity} forecast:", mae[self.quantity])
        return mae[self.quantity]


if __name__ == '__main__':
    sns.set_theme(style="whitegrid")

    # Parse command line arguments
    parser = ArgumentParser(
        prog='regression.py',
        description='Loads csv data, fits a regression model, returns a plot and a precision metric.'
    )
    parser.add_argument('-i', '--input', required=True, help='Path to the input CSV file')
    parser.add_argument('-q', '--quantity', required=True, help='Quantity to model and plot')
    parser.add_argument('-o', '--output', help='Optional path to export the PNG plot')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose messages')
    args = parser.parse_args()

    # Load the data and check that the requested quantity is present
    input_data = load_data(args.input)
    if args.quantity not in input_data.columns:
        print(f"Error: Quantity '{args.quantity}' not found in the data.")
        exit(1)

    # Instatiate the model, do all the processing, and plot the results
    model = RegressionModel(quantity=args.quantity)
    try:
        model = model.fit_and_predict(input_data, train_portion=0.8, print_details=args.verbose)
    except ValueError as e:
        print(f"Error during processing: {e}")
        exit(1)

    model.evaluate()
    model.plot(show_train=True, output_path=args.output)
