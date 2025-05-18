import unittest
import pandas as pd
from time_series import (
    ModelType,
    TimeSeriesModel,
    load_data,
    drop_constant_cols,
    get_endo_exo,
)


class TestModel(unittest.TestCase):
    def test_init(self):
        # Placeholder test to check if the model initializes correctly.
        model = TimeSeriesModel("Consumption", ModelType.LGBM)
        self.assertEqual(model.quantity, 'Consumption')
        self.assertEqual(model.model_type, ModelType.LGBM)
        self.assertEqual(model.verbose, False)


class TestDataProcessing(unittest.TestCase):
    def test_load(self):
        # Placeholder test to check data is loaded as expected.
        data = load_data("data/simple_test_data.csv")
        self.assertEqual(len(data), 3)
        self.assertEqual(type(data.index[0]), pd.Timestamp)

    def test_preproc_constant_column(self):
        # Check that constant columns are dropped correctly.

        # Create a simple DataFrame with a constant column
        data = pd.DataFrame({
            "Constant_column": [1, 1, 1],
            "Non_constant_column": [20, 21, 19]
        })
        data = drop_constant_cols(data, "Non_constant_column")
        self.assertNotIn("Consumption", data.columns)

    def test_endo_exo_split(self):
        # Check that data is correctly split into endogenous and exogenous variables.
        
        data = pd.DataFrame({
            "Endo": [1, 2, 3],
            "Exo": [1, 2, 3]
        })
        data_endo, data_exo = get_endo_exo(data, "Endo")
        self.assertEqual(data_endo.columns, ["Endo"])
        self.assertEqual(data_exo.columns, ["Exo"])


if __name__ == '__main__':
    unittest.main(verbosity=1)