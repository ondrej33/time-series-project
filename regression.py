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


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data.
    
    # TODO: This is just a placeholder to prototype the functionality.
    """

    #data.loc[:, 'Time'] = pd.to_datetime(data['Time'], format='ISO8601', utc=True)

    # TODO: use all columns for the model, this is just a prototype thing
    interesting_quantities = ['Consumption', 'Grid consumption', 'PV generation', 'Battery charging']
    data = data[interesting_quantities]

    # TODO: perhaps do the preprocessing in a more sophisticated and suitable way
    data = data.interpolate()
    data = data.asfreq("30min")

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    data = pd.DataFrame(data_scaled, index=data.index, columns=data.columns)

    return data


def load_data(data_path: str) -> pd.DataFrame:
    """Load the data from a CSV file, parse the time column, and set it as the index."""
    data = pd.read_csv(data_path, sep=";", parse_dates=["Time"], index_col="Time")
    data.index = pd.to_datetime(data.index, format='ISO8601', utc=True)
    return data


class RegressionModel:
    """A wrapper for the regression model.
    
    TODO: This is just a placeholder to prototype the class.
    """
    def __init__(self, data_path: str, quantity: str, output_path: Optional[str] = None):
        # TODO: Decide what should be saved in the class (model, data, etc.)
        self.data = load_data(data_path)
        self.quantity = quantity
        self.output_path = output_path

    def fit_and_predict(self):
        """Fit model to the data, evaluate, and plot the results.
        
        TODO: This is just a placeholder to prototype the functionality.
        """
        # TODO: Add proper preprocessing, model selection, cross-validation, and evaluation

        self.data = preprocess(self.data)
        train_size = int(len(self.data) * 0.8)
        train, test = self.data.iloc[:train_size], self.data.iloc[train_size:]

        # TODO: Check cross-validation scores on the training set
        model = VAR(maxlags=48, ic='aic') # max one day
        cv = ExpandingWindowSplitter(initial_window=524, step_length=48, fh=np.arange(1, 96))
        loss = MeanAbsoluteError()
        results = evaluate(forecaster=model, y=train, cv=cv, scoring=loss, return_model=True)
        print(f"Cross-validation results:\n{results}")

        # Fit model on the full training set
        model_fitted = model.fit(train)

        # Forecast for the test set
        forecast_steps = len(test)
        forecast = model_fitted.predict(fh=np.arange(1, forecast_steps))
        forecast_data = pd.DataFrame(forecast, index=test.index, columns=self.data.columns)

        # TODO: Choose a suitable metric for the time series and selected model 
        mae = np.abs(forecast_data - test).mean()
        print(f"Final MAE scores over all quantities:\n{mae}")

        # TODO: Properly plot actual vs predicted values
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data[self.quantity], label=f"Actual {self.quantity}", linestyle="dashed")
        plt.plot(forecast_data.index, forecast_data[self.quantity], label=f"Forecast {self.quantity}")
        plt.legend()
        plt.title("Normalized VAR Model Forecast")
        if self.output_path is not None:
            plt.savefig(self.output_path)
        else:
            plt.show()


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
    args = parser.parse_args()

    # Instatiate the model instance and do the processing
    model = RegressionModel(
        data_path=args.input, 
        quantity=args.quantity, 
        output_path=args.output
    )
    model.fit_and_predict()
