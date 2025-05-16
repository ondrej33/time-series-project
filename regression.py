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


def preprocess_and_split(
    data: pd.DataFrame, 
    train_portion: float,
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """Preprocess and split the data into training and testing subsets.

    Returns the training and testing data, as well as a scaler for inverse transformation.
    
    # TODO: This is just a placeholder to prototype the functionality.
    """
    # TODO: use all columns for the model, this is just a prototype thing
    interesting_quantities = ['Consumption', 'Grid consumption', 'PV generation', 'Battery charging']
    data = data[interesting_quantities]

    # interpolate missing values 
    data = data.interpolate()
    data = data.asfreq("30min")

    # split the data into training and testing sets so that we can process the
    # data separately and avoid data leakage
    train_size = int(len(data) * train_portion)
    train, test = data.iloc[:train_size], data.iloc[train_size:]

    # simple zero-one scaling to scale the features to the same range
    scaler = MinMaxScaler()
    scaler = scaler.fit(train)
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    train_scaled = pd.DataFrame(train_scaled, index=train.index, columns=train.columns)
    test_scaled = pd.DataFrame(test_scaled, index=test.index, columns=test.columns)
    return train_scaled, test_scaled, scaler


def load_data(data_path: str) -> pd.DataFrame:
    """Load the data from a CSV file, parse the time column, and set it as the index."""
    data = pd.read_csv(data_path, sep=";", parse_dates=["Time"], index_col="Time")
    data.index = pd.to_datetime(data.index, format='ISO8601', utc=True)
    return data


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
        # TODO: Add proper preprocessing, model selection, cross-validation, and evaluation

        self.raw_data = raw_data
        train, test, scaler = preprocess_and_split(self.raw_data, train_portion=train_portion)
        features = train.columns

        # TODO: Check cross-validation scores on the training set
        model = VAR(maxlags=48, ic='aic') # max one day
        cv = ExpandingWindowSplitter(initial_window=480, step_length=48, fh=np.arange(1, 48))
        loss = MeanAbsoluteError()
        results = evaluate(forecaster=model, y=train, cv=cv, scoring=loss, return_model=True)
        if print_details:
            print(f"Cross-validation results:\n{results}")

        # Fit model on the full training set and try forecasting on known data
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
        start_idx = 0 if show_train else len(self.train)        
        plt.plot(self.raw_data.index[start_idx:], self.raw_data[self.quantity][start_idx:], label=f"Actual {self.quantity}", linestyle="dashed")
        plt.plot(forecast_test_scaled.index, forecast_test_scaled[self.quantity], label=f"Forecast {self.quantity}")
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
        print(f"Mean absolute error for normalized {self.quantity} forecast:", mae[self.quantity])
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
        raise ValueError(f"Quantity '{args.quantity}' not found in the data.")

    # Instatiate the model, do all the processing, and plot the results
    model = RegressionModel(quantity=args.quantity)
    model = model.fit_and_predict(input_data, train_portion=0.8, print_details=args.verbose)
    model.evaluate()
    model.plot(show_train=False, output_path=args.output)
