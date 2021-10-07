# Ensure bindings from TestModelFactory working correctly
#
# The data container classes inside TestModelFactory.h have been
# exposed to Python using swig (see TestModelFactory.i).

import os
import sys
import unittest

import numpy as np

thisDir = os.path.dirname(os.path.realpath(__file__))
rr_site_packages = os.path.dirname(os.path.dirname(thisDir))

print(f"Running  {__file__}")

sys.path += [
    rr_site_packages,
]
from roadrunner.testing import TestModelFactory as tmf


class TestModelFactoryTests(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def testDoublePair(self):
        x = tmf._testDoublePair(3.4, 5.6)
        self.assertIsInstance(x, tuple)
        self.assertEqual(x[0], 3.4)

    def testUseVariant(self):
        x = tmf._testVariant()
        self.assertIsInstance(x, float)

    def testVariantMap(self):
        x = tmf._testVariantMap()
        self.assertIsInstance(x, dict)
        self.assertIsInstance(list(x.keys())[0], str)
        self.assertIsInstance(list(x.values())[0], int)

    def testResultMap(self):
        x = tmf._testResultMap()
        self.assertIsInstance(x, dict)

    def testGetSettings(self):
        open_linear_flux = tmf.OpenLinearFlux()
        expected = {'allow_presimulation': False,
                    'auto_moiety_analysis': False,
                    'moiety_conservation': False,
                    'presimulation_time': 5}
        self.assertEqual(expected, open_linear_flux.steadyStateSettings())

    def testSimpleFluxStr(self):
        sf = tmf.SimpleFlux()
        self.assertTrue(sf.str().startswith(r"<?xml"))

    def testSimpleFluxModelName(self):
        self.assertEqual("SimpleFlux", tmf.SimpleFlux().modelName())

    def testSimpleFluxSteadyState(self):
        sf = tmf.SimpleFlux()
        methods = [
            "modelName",
            "timeSeriesResult",
            "steadyState",
            "str",
            "timeSeriesSettings",
            "steadyStateSettings",
        ]
        for i in methods:
            m = getattr(sf, i)
            self.assertTrue(callable(m))

    def testNamesOfClassesExist(self):
        classNames = [
            "SimpleFlux",
            "SimpleFluxManuallyReduced",
            "OpenLinearFlux",
            "Model269",
            "Model28",
            "CeilInRateLaw",
            "FactorialInRateLaw",
            "Venkatraman2010",
            "Brown2004",
        ]
        for clsName in classNames:
            self.assertIn(clsName, dir(tmf))

    def test_polymorphic_factory(self):
        testModel = tmf.TestModelFactory("OpenLinearFlux")
        print("Inside testply", testModel, type(testModel))
        self.assertIsInstance(testModel, tmf.OpenLinearFlux)  # fails if is type TestModel

    def test_string_vector_converts(self):
        self.assertIsInstance(tmf.getAvailableTestModels(), tuple)

    def test_rrDoubleMatrixIsANumpyArray(self):
        """check that the rr::DoubleMatrix is correctly converted into numpy array, much like ls::DoubleMatrix"""
        testModel = tmf.TestModelFactory("SimpleFlux")
        rrDoubleMatrix = testModel.timeSeriesResult()
        self.assertIsInstance(rrDoubleMatrix, np.ndarray)

    def test_rrDoubleMatrixShape(self):
        """check that the rr::DoubleMatrix is correctly converted into numpy array, much like ls::DoubleMatrix"""
        testModel = tmf.TestModelFactory("SimpleFlux")
        rrDoubleMatrix = testModel.timeSeriesResult()
        self.assertEqual((11, 3), rrDoubleMatrix.shape)

    def test_rrDoubleMatrixToNamedArray(self):
        testModel = tmf.TestModelFactory("BimolecularEnd")
        # print(testModel.fullJacobianConc())

    def test_stdComplexZeroImagPart(self):
        """Check swig conversion of ls::Complex that has 0 imag part"""
        cpx = tmf._testStdComplexZeroImagPart()
        self.assertIsInstance(cpx[0], float)
        self.assertAlmostEqual(2, cpx)

    def test_stdComplexNonZeroImagPart(self):
        """Check swig conversion of ls::Complex that has 0 imag part"""
        cpx = tmf._testStdComplexNonZeroImagPart()
        self.assertIsInstance(cpx[0], complex)
        self.assertAlmostEqual(3, cpx.real)
        self.assertAlmostEqual(4, cpx.imag)

    def test_doubleVectorToNumpyArray_CorrectType(self):
        arr = tmf._testDoubleVectorTo1DNumpyArray()
        self.assertIsInstance(arr, np.ndarray)

    def test_doubleVectorToNumpyArray_CorrectValues(self):
        actual = tmf._testDoubleVectorTo1DNumpyArray()
        expected = np.array([0.1, 0.2, 0.3])
        print(expected)
        print(actual)
        self.assertTrue((expected == actual).all())



    def test_testMatrix3D_index3x2x3(self):
        (idx, mat, rownames, colnames) = tmf._testMatrix3D_3x2x3()
        expected = np.array([0, 3.5, 7.9])
        self.assertTrue((idx == expected).all())

    def test_testMatrix3D_data3x2x3(self):
        (idx, mat, rownames, colnames) = tmf._testMatrix3D_3x2x3()
        expected = np.array(
            [[[0.1, 0.2, 0.3],
              [0.4, 0.5, 0.6]],

             [[0.7, 0.8, 0.9],
              [1., 1.1, 1.2]],

             [[1.3, 1.4, 1.5],
              [1.6, 1.7, 1.8]]]
        )
        self.assertTrue((expected == mat).all())

    def test_testMatrix3D_index2x3x4(self):
        (idx, mat, rownames, colnames) = tmf._testMatrix3D_2x3x4()
        expected = np.array([0, 4.23])
        self.assertTrue((idx == expected).all())

    def test_testMatrix3D_data2x3x4(self):
        (idx, mat, rownames, colnames) = tmf._testMatrix3D_2x3x4()
        expected = np.array(
            [
                [
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [8, 9, 10, 11]
                ],
                [
                    [12, 13, 14, 15],
                    [16, 17, 18, 19],
                    [20, 21, 22, 23],
                ]
            ]
        )
        self.assertTrue((expected == mat).all())

    def test_testMatrix3D_index4x3x2(self):
        (idx, mat, rownames, colnames) = tmf._testMatrix3D_4x3x2()
        expected = np.array([0, 6, 12, 18])
        self.assertTrue((idx == expected).all())

    def test_testMatrix3D_data4x3x2(self):
        (idx, mat, rownames, colnames) = tmf._testMatrix3D_4x3x2()
        expected = np.array(
            [
                [
                    [0, 1],
                    [2, 3],
                    [4, 5]
                ],
                [
                    [6, 7],
                    [8, 9],
                    [10, 11],
                ],
                [
                    [12, 13],
                    [14, 15],
                    [16, 17],
                ],
                [
                    [18, 19],
                    [20, 21],
                    [22, 23],
                ]
            ]
        )
        self.assertTrue((expected == mat).all())

    def test_testMatrix3D_index3x4x2(self):
        (idx, mat, rownames, colnames) = tmf._testMatrix3D_3x4x2()
        expected = np.array([0, 6, 12])
        self.assertTrue((idx == expected).all())

    def test_testMatrix3D_data3x4x2(self):
        (idx, mat, rownames, colnames) = tmf._testMatrix3D_3x4x2()
        expected = np.array(
            [
                [
                    [0, 1],
                    [2, 3],
                    [4, 5],
                    [6, 7]
                ],
                [
                    [8, 9],
                    [10, 11],
                    [12, 13],
                    [14, 15],
                ],
                [
                    [16, 17],
                    [18, 19],
                    [20, 21],
                    [22, 23],
                ]
            ]
        )
        self.assertTrue((expected == mat).all())

    def test_index_calculation(self):
        """Figure our linear index of a 3D matrix.

        f(x, y, z) = a linear function from 0, 1, ..., x*y*z

        Technically not a test, but this calculation is quite painful
        to workout so it is left as a reminder
        :return:
        """
        xMax = 5
        yMax = 4
        zMax = 2

        ans = []
        for z in range(zMax):
            for y in range(yMax):
                for x in range(xMax):
                    ans.append(x + y * xMax + z * xMax * yMax)
        self.assertEqual([i for i in range(xMax*yMax*zMax)], ans)

    def test_python_str_to_cxx_str(self):
        # the input argument does a round trip from python to C and back
        roundTrip = tmf._testPythonStringToCxxRoundTrip("Python String")
        self.assertIsInstance(roundTrip, str)
        self.assertEqual("Python String", roundTrip)

    def test_list_of_strings_by_value(self):
        l = ["first", "second", "third"]
        l = tmf._testAddElementToStringVec(l, "fourth")
        self.assertEqual(l, ("first", "second", "third", "fourth"))












