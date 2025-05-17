import unittest
import pandas as pd
from time_series import ModelType, TimeSeriesModel, load_data, drop_constant_cols


class TestModel(unittest.TestCase):
    def test_init(self):
        model = TimeSeriesModel("Consumption", ModelType.LGBM)
        self.assertEqual(model.quantity, 'Consumption')
        self.assertEqual(model.model_type, ModelType.LGBM)
        self.assertEqual(model.verbose, False)


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

        # check that the constant column is removed
        data = drop_constant_cols(data, "Non_constant_column")
        self.assertNotIn("Consumption", data.columns)


if __name__ == '__main__':
    unittest.main(verbosity=2)