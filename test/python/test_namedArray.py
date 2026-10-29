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


class NamedArrayCompatibilityTests(unittest.TestCase):
    """Behavior of NamedArray as seen through the compiled module."""

    def simulation(self):
        from roadrunner import RoadRunner
        from roadrunner.tests import TestModelFactory as tmf
        m = RoadRunner(tmf.SimpleFlux().str())
        return m, m.simulate(0, 10, 11)

    def test_class_is_reachable_from_old_and_new_locations(self):
        import roadrunner.named_array
        import roadrunner._roadrunner
        self.assertIs(roadrunner._roadrunner.NamedArray, roadrunner.named_array.NamedArray)
        self.assertEqual(NamedArray.__module__, "roadrunner._roadrunner")
        self.assertTrue(issubclass(NamedArray, np.ndarray))

    def test_simulation_result_type_and_dtype(self):
        m, sim = self.simulation()
        self.assertIsInstance(sim, NamedArray)
        self.assertEqual(sim.dtype, np.float64)
        self.assertEqual(sim.rownames, [])
        self.assertEqual(sim.colnames, ['time', '[S1]', '[S2]'])

    def test_simulation_result_index_by_column_name(self):
        m, sim = self.simulation()
        col = sim['time']
        self.assertEqual(col.shape, (11,))
        np.testing.assert_array_equal(col, sim[:, 0])
        np.testing.assert_allclose(col, np.linspace(0, 10, 11))
        np.testing.assert_array_equal(sim['[S2]'], sim[:, 2])

    def test_simulation_result_str_shows_names_and_every_row(self):
        m, sim = self.simulation()
        text = str(sim)
        for name in sim.colnames:
            self.assertIn(name, text)
        self.assertGreaterEqual(len(text.splitlines()), 11)
        self.assertEqual(text, repr(sim))

    def test_simulation_result_is_independent_of_later_simulations(self):
        m, sim = self.simulation()
        expected = np.array(sim)
        m.reset()
        m.simulate(0, 20, 21)
        self.assertEqual(sim.shape, (11, 3))
        np.testing.assert_array_equal(sim, expected)

    def test_simulation_result_is_writeable_without_changing_the_model(self):
        m, sim = self.simulation()
        before = m.model.getFloatingSpeciesConcentrations().copy()
        sim[:, 1:] = 12345.0
        np.testing.assert_array_equal(m.model.getFloatingSpeciesConcentrations(), before)

    def test_pickle_round_trip_of_simulation_result(self):
        m, sim = self.simulation()
        loaded = pickle.loads(pickle.dumps(sim))
        self.assertIsInstance(loaded, NamedArray)
        self.assertEqual(loaded.colnames, sim.colnames)
        self.assertEqual(loaded.rownames, sim.rownames)
        np.testing.assert_array_equal(loaded, sim)

    def test_pickle_refers_to_the_old_module_path(self):
        n = np.zeros((2, 3)).view(NamedArray)
        self.assertIn(b"roadrunner._roadrunner", pickle.dumps(n))

    def test_pickle_written_in_the_original_format_still_loads(self):
        values = np.arange(6, dtype=np.float64)

        class OriginalPickle:
            def __reduce__(self):
                state = {
                    'array': values.tobytes(),
                    'nDims': 2,
                    'dim1': 2,
                    'dim2': 3,
                    'rownames': ['R1', 'R2'],
                    'colnames': ['C1', 'C2', 'C3'],
                    '_pickle_version': 5,
                }
                return NamedArray, ((2, 3),), state

        loaded = pickle.loads(pickle.dumps(OriginalPickle()))
        self.assertIsInstance(loaded, NamedArray)
        np.testing.assert_array_equal(loaded, values.reshape(2, 3))
        self.assertEqual(loaded.rownames, ['R1', 'R2'])
        self.assertEqual(loaded.colnames, ['C1', 'C2', 'C3'])

    def test_jacobian_has_matching_row_and_column_names(self):
        from roadrunner import RoadRunner
        from roadrunner.tests import TestModelFactory as tmf
        m = RoadRunner(tmf.SimpleFlux().str())
        j = m.getFullJacobian()
        self.assertIsInstance(j, NamedArray)
        self.assertEqual(j.shape[0], j.shape[1])
        self.assertEqual(len(j.rownames), j.shape[0])
        self.assertEqual(j.rownames, j.colnames)
        np.testing.assert_array_equal(j[j.colnames[0]], j[:, 0])

    def test_slices_and_ufuncs_stay_named_arrays(self):
        m, sim = self.simulation()
        self.assertIsInstance(sim[2:5, :], NamedArray)
        self.assertIsInstance(sim * 2, NamedArray)
        self.assertEqual((sim * 2).shape, sim.shape)
        self.assertIsInstance(np.asarray(sim), np.ndarray)
        self.assertNotIsInstance(np.asarray(sim), NamedArray)


if __name__ == "__main__":
    unittest.main()
