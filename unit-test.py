import unittest
import pandas as pd
from regression import RegressionModel, load_data, preprocess_and_split


class TestModel(unittest.TestCase):
    def test_init(self):
        model = RegressionModel("Consumption")
        self.assertEqual(model.quantity, 'Consumption')


class TestDataProcessing(unittest.TestCase):
    def test_load(self):
        data = load_data("simple_test_data.csv")
        self.assertEqual(len(data), 3)

    def test_preproc_constant_column(self):
        # Create a simple DataFrame with a constant column
        data = pd.DataFrame({
            "Time": ["2023-01-01T00:00:00Z", "2023-01-01T00:30:00Z", "2023-01-01T01:00:00Z"],
            "Constant_column": [1, 1, 1],
            "Non_constant_column": [20, 21, 19]
        })
        data["Time"] = pd.to_datetime(data["Time"], format='ISO8601', utc=True)
        data.set_index("Time", inplace=True)

        # check that the constant column is removed
        train, _, _ = preprocess_and_split(data, 0.5, "Non_constant_column")
        self.assertNotIn("Consumption", train.columns)


if __name__ == '__main__':
    unittest.main(verbosity=2)