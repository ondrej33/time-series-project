import unittest
from regression import RegressionModel

example_data_path = "data.csv"

class TestModel(unittest.TestCase):
    def test_init(self):
        model = RegressionModel("simple_test_data.csv", "Consumption")
        self.assertEqual(model.quantity, 'Consumption')
        self.assertEqual(len(model.data), 3)

if __name__ == '__main__':
    unittest.main(verbosity=2)