import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("rrplugins"), "rrplugins is not installed")
class DataSeriesFromNumPyTests(unittest.TestCase):
    """rrplugins reads numpy and roadrunner results into its own data series."""

    def setUp(self):
        import rrplugins
        self.tel = rrplugins

    def headers(self, series):
        headers = series.getColumnHeaders()
        if isinstance(headers, str):
            headers = headers.split(",")
        return list(headers)

    def test_structured_array_goes_through_view(self):
        dtype = np.dtype([("time", "f8"), ("A", "f8"), ("B", "f8")])
        structured = np.zeros(4, dtype=dtype)
        structured["time"] = [0.0, 1.0, 2.0, 3.0]
        structured["A"] = [10.0, 11.0, 12.0, 13.0]
        structured["B"] = [20.0, 21.0, 22.0, 23.0]
        series = self.tel.DataSeries.fromNumPy(structured)
        self.assertEqual((series.rows, series.cols), (4, 3))
        self.assertEqual(self.headers(series), ["time", "A", "B"])
        np.testing.assert_array_equal(series.toNumPy(), structured.view((float, 3)))

    def test_plain_array_gets_made_up_column_names(self):
        values = np.arange(6, dtype=float).reshape(3, 2)
        series = self.tel.DataSeries.fromNumPy(values)
        self.assertEqual((series.rows, series.cols), (3, 2))
        self.assertEqual(self.headers(series), ["x0", "x1"])
        np.testing.assert_array_equal(series.toNumPy(), values)

    def test_roadrunner_simulation_result_keeps_its_column_names(self):
        from roadrunner import RoadRunner
        from roadrunner.tests import TestModelFactory as tmf
        sim = RoadRunner(tmf.SimpleFlux().str()).simulate(0, 10, 11)
        series = self.tel.DataSeries.fromNumPy(sim)
        self.assertEqual((series.rows, series.cols), sim.shape)
        self.assertEqual(self.headers(series), sim.colnames)
        np.testing.assert_allclose(series.toNumPy(), np.asarray(sim))

    def test_one_dimensional_input_is_rejected(self):
        with self.assertRaises((ValueError, IndexError)):
            self.tel.DataSeries.fromNumPy(np.arange(5, dtype=float))


if __name__ == "__main__":
    unittest.main()
