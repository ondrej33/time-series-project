import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from statsmodels.tsa.api import VAR
from sklearn.preprocessing import MinMaxScaler


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


class RegressionModel:
    """A wrapper for the regression model.
    
    TODO: This is just a placeholder to prototype the class.
    """
    def __init__(self, data_file_path: str, quantity: str):
        # TODO: Decide what should be saved in the class (model, data, etc.)
        self.data = pd.read_csv(data_file_path, sep=";", parse_dates=["Time"], index_col="Time")
        self.data.index = pd.to_datetime(self.data.index, format='ISO8601', utc=True)
        self.quantity = quantity

    def fit_data(self):
        """Fit model to the data, evaluate, and plot the results.
        
        TODO: This is just a placeholder to prototype the functionality.
        """
        # TODO: Add proper preprocessing, model selection, cross-validation, and evaluation

        # TODO: Suitable preprocessing and splitting for the model
        # TODO: Cross-validation
        self.data = preprocess(self.data)
        train_size = int(len(self.data) * 0.8)
        train, test = self.data.iloc[:train_size], self.data.iloc[train_size:]

        # TODO: Choose a suitable model for the model
        model = VAR(train)
        lag_order = model.select_order(maxlags=96) # max two days
        model_fitted = model.fit(lag_order.aic)

        forecast_steps = len(test)
        forecast = model_fitted.forecast(train.values[-lag_order.aic:], steps=forecast_steps)
        forecast_data = pd.DataFrame(forecast, index=test.index, columns=self.data.columns)

        # TODO: Choose a suitable metric for the time series and selected model 
        mae = np.abs(forecast_data - test).mean()
        print(f"Mean Absolute Error:\n{mae}")

        # TODO: Properly plot actual vs predicted values
        plt.figure(figsize=(12,6))
        plt.plot(self.data.index, self.data[self.quantity], label=f"Actual {self.quantity}", linestyle="dashed")
        plt.plot(forecast_data.index, forecast_data[self.quantity], label=f"Forecast {self.quantity}")
        plt.legend()
        plt.title("VAR Model Forecast")
        plt.show()


if __name__ == '__main__':
    sns.set_theme(style="whitegrid")

    # Parse command line arguments
    parser = ArgumentParser(
                prog='regression.py',
                description='Loads csv data, fits a regression model, returns a plot and a precision metric.')
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-q', '--quantity', required=True)
    # TODO: Add arguments (output file, model type, etc.)
    args = parser.parse_args()

    # Instatiate the model instance and do the processing
    model = RegressionModel(data_file_path=args.input, quantity=args.quantity)
    model.fit_data()
