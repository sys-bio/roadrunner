import os
import unittest

import sys

thisDir = os.path.dirname(os.path.realpath(__file__))
rr_site_packages = os.path.dirname(os.path.dirname(thisDir))

sys.path += [
    rr_site_packages,
]

try:
    from roadrunner.roadrunner import (
        RoadRunner, Integrator, SteadyStateSolver,
        ExecutableModel, RoadRunnerOptions
    )
    import roadrunner
except ImportError:
    from roadrunner import (
        RoadRunner, Integrator, SteadyStateSolver,
        ExecutableModel, RoadRunnerOptions
    )
    import roadrunner

from roadrunner.testing import SettingTestsSwigAPI


class SettingTests(unittest.TestCase):
    """
    A Setting (aka Variant) is the roadrunner construct for
    containing multiple types and its used for storing
    roadrunner options. Since we deliberately convert
    Setting's to their corresponding type in Python,
    and this conversion should happen seamlessly,
    the Setting object itself is not exposed to the Python
    level. We must therefore test it indirectly.
    """

    def setUp(self):
        self.s = SettingTestsSwigAPI.SettingTestsSwigAPI()

    def testCxxToPy_char(self):
        self.assertEqual(self.s.getCharTypeAsSetting(), "c")

    def testCxxToPy_uchar(self):
        self.assertEqual(self.s.getUCharTypeAsSetting(), "d")

    def testCxxToPy_float(self):
        self.assertAlmostEqual(self.s.getFloatTypeAsSetting(), 0.1234)

    def testCxxToPy_double(self):
        self.assertAlmostEqual(self.s.getDoubleTypeAsSetting(), 0.5678)

    def testCxxToPy_int32(self):
        self.assertEqual(self.s.getInt32TypeAsSetting(), -1)

    def testCxxToPy_uint32(self):
        self.assertEqual(self.s.getUint32TypeAsSetting(), 1)

    def testCxxToPy_int64(self):
        self.assertEqual(self.s.getInt64TypeAsSetting(), -2)

    def testCxxToPy_uint64(self):
        self.assertEqual(self.s.getUint64TypeAsSetting(), 2)

    def testCxxToPy_doubleVector(self):
        self.assertEqual(self.s.getDoubleVectorTypeAsSetting(), [0.1, 0.2, 0.3])

    def testCxxToPy_doubleVectorIsList(self):
        """
        We might want to consider converting std::vector
        into a numpy array instead...
        :return:
        """
        self.assertIsInstance(self.s.getDoubleVectorTypeAsSetting(), list)

    def testCxxToPy_string(self):
        self.assertEqual(self.s.getStringTypeAsSetting(), "I'm a string")

    def testPyToCxx_char(self):
        self.s.setCharType("t")
        self.assertEqual(self.s.getCharTypeAsSetting(), "t")

    @unittest.skip("Not supported. Conversion from Python string to "
                   "C++ will always go to std::string, not char or uchar")
    def testPyToCxx_uchar(self):
        pass

    def testPyToCxx_string(self):
        self.s.setStringType("I'm a Python string")
        self.assertEqual(self.s.getStringTypeAsSetting(), "I'm a Python string")

    def testPyToCxx_bool(self):
        self.s.setBoolType(False)
        self.assertFalse(self.s.getBoolTypeAsSetting())

    def testPyToCxx_int32(self):
        self.s.setInt32Type(9876)
        self.assertEqual(self.s.getInt32TypeAsSetting(), 9876)

    def testPyToCxx_uint32(self):
        self.s.setUint32Type(8476)
        self.assertEqual(self.s.getUint32TypeAsSetting(), 8476)

    def testPyToCxx_int64(self):
        self.s.setInt64Type(9287646)
        self.assertEqual(self.s.getInt64TypeAsSetting(), 9287646)

    def testPyToCxx_uint64(self):
        self.s.setUint64Type(998763552)
        self.assertEqual(self.s.getUint64TypeAsSetting(), 998763552)

    def testPyToCxx_float(self):
        self.s.setFloatType(0.8576635)
        self.assertEqual(self.s.getFloatTypeAsSetting(), 0.8576635)

    def testPyToCxx_double(self):
        self.s.setDoubleType(8.776535)
        self.assertEqual(self.s.getDoubleTypeAsSetting(), 8.776535)

    def testPyToCxx_doubleVector(self):
        self.s.setDoubleVectorType([0.6, 8.3, 76.2])
        self.assertEqual(self.s.getDoubleVectorTypeAsSetting(), [0.6, 8.3, 76.2])

    def test_get_stringType_from_settings_map(self):
        self.assertEqual("I'm a string", self.s.getValueFromSettings("stringType"))

    def test_get_boolType_from_settings_map(self):
        self.assertEqual(True, self.s.getValueFromSettings("boolType"))

    def test_get_int32Type_from_settings_map(self):
        self.assertEqual(-1, self.s.getValueFromSettings("int32Type"))

    def test_get_uint32Type_from_settings_map(self):
        self.assertEqual(1, self.s.getValueFromSettings("uint32Type"))

    def test_get_int64Type_from_settings_map(self):
        self.assertEqual(-2, self.s.getValueFromSettings("int64Type"))

    def test_get_uint64Type_from_settings_map(self):
        self.assertEqual(2, self.s.getValueFromSettings("uint64Type"))

    def test_get_floatType_from_settings_map(self):
        self.assertEqual(0.1234, self.s.getValueFromSettings("floatType"))

    def test_get_doubleType_from_settings_map(self):
        self.assertEqual(0.5678, self.s.getValueFromSettings("doubleType"))

    def test_get_charType_from_settings_map(self):
        self.assertEqual("c", self.s.getValueFromSettings("charType"))

    def test_get_uCharType_from_settings_map(self):
        self.assertEqual("d", self.s.getValueFromSettings("uCharType"))

    def test_get_doubleVectorType_from_settings_map(self):
        self.assertEqual([0.1, 0.2, 0.3], self.s.getValueFromSettings("doubleVectorType"))

    def test_set_stringType_in_settings_map(self):
        self.s.setValueInSettings("stringType", "A New String")
        self.assertEqual("A New String", self.s.getValueFromSettings("stringType"))

    def test_set_boolType_in_settings_map(self):
        self.s.setValueInSettings("boolType", False)
        self.assertEqual(False, self.s.getValueFromSettings("boolType"))

    def test_set_int32Type_in_settings_map(self):
        self.s.setValueInSettings("int32Type", 345634)
        self.assertEqual(345634, self.s.getValueFromSettings("int32Type"))

    def test_set_uint32Type_in_settings_map(self):
        self.s.setValueInSettings("uint32Type", 887)
        self.assertEqual(887, self.s.getValueFromSettings("uint32Type"))

    def test_set_int64Type_in_settings_map(self):
        self.s.setValueInSettings("int64Type", 98735)
        self.assertEqual(98735, self.s.getValueFromSettings("int64Type"))

    def test_set_uint64Type_in_settings_map(self):
        self.s.setValueInSettings("uint64Type", 254353)
        self.assertEqual(254353, self.s.getValueFromSettings("uint64Type"))

    def test_set_floatType_in_settings_map(self):
        self.s.setValueInSettings("floatType", 9.7643)
        self.assertEqual(9.7643, self.s.getValueFromSettings("floatType"))

    def test_set_doubleType_in_settings_map(self):
        self.s.setValueInSettings("doubleType", 8.65432)
        self.assertEqual(8.65432, self.s.getValueFromSettings("doubleType"))

    def test_set_charType_in_settings_map(self):
        self.s.setValueInSettings("charType", "h")
        self.assertEqual("h", self.s.getValueFromSettings("charType"))

    def test_set_uCharType_in_settings_map(self):
        self.s.setValueInSettings("uCharType", "o")
        self.assertEqual("o", self.s.getValueFromSettings("uCharType"))

    def test_set_doubleVectorType_in_settings_map(self):
        self.s.setValueInSettings("doubleVectorType", [9.8, 6.5])
        self.assertEqual([9.8, 6.5], self.s.getValueFromSettings("doubleVectorType"))
