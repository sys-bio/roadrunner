import numpy as np
import sys

import pickle
import unittest
import roadrunner
from roadrunner._roadrunner import NamedArray

print(roadrunner.__file__)


class NamedArrayTests(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_NamedArray_is_part_of__roadrunner_module(self):
        """
        the tp_name field of NamedArray_Type needed to be
        changed to roadrunner._roadrunner.NamedArray rather
        than NamedArray so that the __module__attribute
        pointed to the correct location and serialization.dumps
        looks in the right place for NamedArray
        :return:
        """
        from roadrunner._roadrunner import NamedArray
        self.assertTrue(NamedArray.__module__, "roadrunner._roadrunner")

    def test_init_from_constructor(self):
        """

        :return:
        """
        n = NamedArray((2, 3))
        self.assertIsInstance(n, NamedArray)
        # note getrefcount itself bumps the refcount up by 1
        self.assertGreaterEqual(2, sys.getrefcount(n))
        self.assertEqual(n.rownames, [])

    def test_init_from_view_casting(self):
        arr = np.ndarray((2, 3))
        for i in range(2):
            for j in range(3):
                arr[i, j] = 3 * i + j
        print(arr)
        self.assertIsInstance(arr, np.ndarray)
        n = arr.view(NamedArray)
        self.assertGreaterEqual(2, sys.getrefcount(n))  # -1 for the reference used by getrefcount
        self.assertIsInstance(n, NamedArray)
        print(n)

    def test_init_from_template(self):
        n = NamedArray((2, 3))
        for i in range(2):
            for j in range(3):
                n[i, j] = 3 * i + j
        print(n)
        self.assertGreaterEqual(2, sys.getrefcount(n))  # -1 for the reference used by getrefcount
        v = n[1:, :]
        print(v)
        self.assertGreaterEqual(2, sys.getrefcount(v))  # -1 for the reference used by getrefcount
        self.assertIsInstance(n, NamedArray)
        self.assertEqual((1, 3), v.shape)

    @unittest.skip("This operation does not work. See comments in PyUtils::NamedArrayObject_Finalize_FromNamedArray")
    def test_init_from_template_with_rownames(self):
        n = NamedArray((2, 3))
        n.rownames = ['R1', 'R2']
        self.assertGreaterEqual(2, sys.getrefcount(n))  # -1 for the reference used by getrefcount
        v = n[1:, :]
        self.assertEqual((1, 3), v.shape)
        self.assertEqual(['R2'], v.rownames)

    @unittest.skip("This operation does not work. See comments in PyUtils::NamedArrayObject_Finalize_FromNamedArray")
    def test_init_from_template_with_rownames_ref_count(self):
        nrow = 4
        ncol = 5
        n = NamedArray((nrow, ncol))
        for i in range(nrow):
            for j in range(ncol):
                n[i, j] = ncol * i + j
        n.rownames = ['R1', 'R2', 'R3', 'R4']
        print('\n')
        print(n)
        self.assertGreaterEqual(2, sys.getrefcount(n.rownames))  # -1 for the reference used by getrefcount
        v = n[2:, :]
        print(v)
        self.assertGreaterEqual(2, sys.getrefcount(v.rownames))  # -1 for the reference used by getrefcount
        self.assertEqual((2, 5), v.shape)
        self.assertEqual(['R3', 'R4'], v.rownames)

    @unittest.skip("This operation does not work. See comments in PyUtils::NamedArrayObject_Finalize_FromNamedArray")
    def test_init_from_template_with_colnames_ref_count(self):
        n = NamedArray((2, 3))
        for i in range(2):
            for j in range(3):
                n[i, j] = 3 * i + j
        n.colnames = ['C1', 'C2', 'C3']
        self.assertGreaterEqual(2, sys.getrefcount(n.colnames))  # -1 for the reference used by getrefcount
        v = n[:, 1:]
        print(v)
        self.assertGreaterEqual(2, sys.getrefcount(v.colnames))  # -1 for the reference used by getrefcount
        self.assertEqual((2, 2), v.shape)
        self.assertEqual(['C1', 'C2'], v.colnames)

    @unittest.skip("AttributeError: 'numpy.ndarray' object has no attribute 'colnames'"
                   "This isn't supported by NamedArray")
    def test_column_headings_after_vstack(self):
        n1 = NamedArray((3, 4))
        n1.colnames = ['C1', 'C2', 'C3']
        n2 = NamedArray((3, 4))
        n2.colnames = ['C1', 'C2', 'C3']
        n3 = np.vstack([n1, n2])
        self.assertEqual(['C1', 'C2', 'C3'], n3.colnames)

    def test_rownames_when_empty(self):
        n = NamedArray((2, 3))
        self.assertEqual([], n.rownames)

    def test_colnames_when_empty(self):
        n = NamedArray((2, 3))
        self.assertEqual([], n.colnames)

    def test_assigning_rownames(self):
        n = NamedArray((2, 3))
        n.rownames = ["R1", "R2"]
        print(n.rownames, ["R1", "R2"])
        self.assertEqual(n.rownames, ["R1", "R2"])

    def test_assigning_colnames(self):
        n = NamedArray((2, 3))
        n.rownames = ["C1", "C2", 'C3']
        self.assertEqual(n.rownames, ["C1", "C2", 'C3'])

    def test_rowname_refcount(self):
        n = NamedArray((2, 3))
        self.assertGreaterEqual(2, sys.getrefcount(n.rownames))  # getrefcount itself adds 1 to ref count

    def test_colname_refcount(self):
        n = NamedArray((2, 3))
        self.assertGreaterEqual(2, sys.getrefcount(n.colnames))  # getrefcount itself adds 1 to ref count

    def test_getstate_isdict(self):
        n = NamedArray((2, 3))
        self.assertIsInstance(n.__getstate__(), dict)

    def test_getstate_rownames_when_empty(self):
        n = NamedArray((2, 3))
        state = n.__getstate__()
        print(state)
        self.assertEqual(state['rownames'], [])

    def test_getstate_rownames_when_full(self):
        n = NamedArray((2, 3))
        n.rownames = [f'R{i}' for i in range(1, 4)]
        state = n.__getstate__()
        print(state)
        self.assertEqual(state['rownames'], ['R1', 'R2', 'R3'])

    def test_getstate_colnames(self):
        n = NamedArray((2, 3))
        n.colnames = [f'C{i}' for i in range(1, 5)]
        state = n.__getstate__()
        print(state['colnames'])
        self.assertEqual(state['colnames'], [f'C{i}' for i in range(1, 5)])

    def test_getstate_array(self):
        n = NamedArray((2, 3))
        state = n.__getstate__()
        self.assertIsInstance(state['array'], bytes)

    def test_reduce_callable(self):
        n = NamedArray((2, 3))
        c, args, state, _, _ = n.__reduce_ex__(5)
        self.assertTrue(callable(c))

    def test_reduce_args(self):
        n = NamedArray((2, 3))
        c, args, state, _, _ = n.__reduce_ex__(5)
        self.assertEqual(args, ((2, 3),))

    def test_reduce_state(self):
        n = NamedArray((2, 3))
        c, args, state, _, _ = n.__reduce_ex__(5)
        self.assertEqual(
            sorted(list(state.keys())),
            ['_pickle_version', 'array', 'colnames', 'dim1', 'dim2', 'nDims', 'rownames']
        )

    def test_reduce_args_work_with_callable(self):
        n = NamedArray((2, 3))
        c, args, state, _, _ = n.__reduce_ex__(5)
        initialized = c(*args)
        self.assertIsInstance(initialized, NamedArray)
        self.assertEqual(initialized.shape, (2, 3))

    def test_reduce_and_setstate(self):
        n = NamedArray((2, 3))
        c, args, state, _, _ = n.__reduce_ex__(5)
        newArray = c(*args)
        newArray.__setstate__(state)
        # make sure we still have a NamedArray after setstate
        self.assertIsInstance(newArray, NamedArray)
        self.assertEqual(newArray.shape, (2, 3))

    def test_getstate_and_setstate(self):
        n = NamedArray((2, 3))
        n[(0, 0)] = 9
        pickled = NamedArray(n.shape)
        self.assertNotEqual(id(n), id(pickled))
        state = n.__getstate__()
        self.assertIsInstance(state, dict)
        pickled.__setstate__(state)
        self.assertAlmostEqual(pickled[(0, 0)], 9)
        #self.assertAlmostEqual(n[(0, 0)], 9)

    def test_pickle_dumps_no_row_or_col(self):
        n = np.zeros((2, 3)).view(NamedArray)
        self.assertIsInstance(n, NamedArray)
        b = pickle.dumps(n)
        # can't check content here, only that we have successfully
        # acquired a bytes object
        self.assertIsInstance(b, bytes)

    def test_pickle_dumps_cols(self):
        n = np.zeros((2, 3)).view(NamedArray)
        n.colnames = ['C1', 'C2', 'C3']
        self.assertIsInstance(n, NamedArray)
        b = pickle.dumps(n)
        # can't check content here, only that we have successfully
        # acquired a bytes object
        self.assertIsInstance(b, bytes)

    def test_pickle_dumps_rows(self):
        n = np.zeros((2, 3)).view(NamedArray)
        n.rownames = ['R1', 'R2']
        self.assertIsInstance(n, NamedArray)
        b = pickle.dumps(n)
        # can't check content here, only that we have successfully
        # acquired a bytes object
        self.assertIsInstance(b, bytes)

    def test_equality_still_works(self):
        n = np.zeros((2, 3)).view(NamedArray)
        n2 = np.zeros((2, 3)).view(NamedArray)
        self.assertTrue((n == n2).all())

    def test_pickle_loads(self):
        n = np.zeros((2, 3)).view(NamedArray)
        for i in range(2):
            for j in range(3):
                n[i, j] = i * 3 + j
        b = pickle.dumps(n)
        l = pickle.loads(b)
        self.assertTrue((l == n).all())

    def test_pickle_loads_with_rownames(self):
        n = np.zeros((2, 3)).view(NamedArray)
        for i in range(2):
            for j in range(3):
                n[i, j] = i * 3 + j
        n.rownames = ['R1', 'R2']
        b = pickle.dumps(n)
        l = pickle.loads(b)
        self.assertEqual(l.rownames, ['R1', 'R2'])
        self.assertTrue((n == l).all())

    def test_pickle_loads_with_colnames(self):
        n = np.zeros((2, 3)).view(NamedArray)
        for i in range(2):
            for j in range(3):
                n[i, j] = i * 3 + j
        n.colnames = ['C1', 'C2', 'C3']
        print(n.colnames)
        b = pickle.dumps(n)
        l = pickle.loads(b)
        self.assertEqual(l.colnames, ['C1', 'C2', 'C3'])
        self.assertTrue((n == l).all())

    def test_df_from_simulation(self):
        from roadrunner import RoadRunner
        from roadrunner.tests import TestModelFactory as tmf
        m = RoadRunner(tmf.SimpleFlux().str())
        sim = m.simulate(0, 10, 11)
        self.assertEqual((11, 3), sim.shape)
        self.assertEqual(sim.colnames, ['time', '[S1]', '[S2]'])

    def test_df_from_steadystate(self):
        from roadrunner import RoadRunner
        from roadrunner.tests import TestModelFactory as tmf
        m = RoadRunner(tmf.SimpleFlux().str())
        m.steadyState()
        mat = m.getSteadyStateValuesNamedArray()
        self.assertEqual((1, 2), mat.shape)
        self.assertEqual(mat.colnames, [ '[S1]', '[S2]'])

    def test_viewcast_NamedArray_to_ndarray(self):
        n = NamedArray((2, 3))
        for i in range(2):
            for j in range(3):
                n[i, j] = 3*i + j
        arr = n.view(np.ndarray)
        self.assertIsInstance(arr, np.ndarray)

    def test_viewcast_ndarray_to_NamedArray(self):
        arr = NamedArray((2, 3))
        for i in range(2):
            for j in range(3):
                arr[i, j] = 3*i + j
        n = arr.view(NamedArray)
        self.assertIsInstance(n, NamedArray)

    def test_viewcast_NamedArray_to_masked_array(self):
        n = NamedArray((2, 3))
        for i in range(2):
            for j in range(3):
                n[i, j] = 3*i + j
        arr = n.view(np.ma.MaskedArray)
        self.assertIsInstance(arr, np.ma.MaskedArray)

    def test_viewcast_masked_array_to_NamedArray(self):
        arr = np.ndarray((2,3))
        for i in range(2):
            for j in range(3):
                arr[i, j] = 3*i + j
        m = arr.view(np.ma.MaskedArray)
        n = m.view(NamedArray)
        self.assertIsInstance(n, NamedArray)

if __name__ == "__main__":
    unittest.main()
