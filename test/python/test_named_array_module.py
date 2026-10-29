"""Tests for wrappers/Python/roadrunner/named_array.py.

The module is loaded straight from the source tree, so these run without
building the compiled extension.
"""
import importlib.util
import pathlib
import pickle
import sys
import unittest

import numpy as np

_SOURCE = pathlib.Path(__file__).resolve().parents[2] / "wrappers" / "Python" / "roadrunner" / "named_array.py"
if not _SOURCE.exists():
    raise unittest.SkipTest("named_array.py source not found; these tests run from the source tree")

_spec = importlib.util.spec_from_file_location("named_array_under_test", _SOURCE)
named_array = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(named_array)

NamedArray = named_array.NamedArray
matrix_to_array = named_array.matrix_to_array

# Pickle finds the class through its module, so point it at this test module.
NamedArray.__module__ = __name__


def _root_base(array):
    while isinstance(array, np.ndarray) and array.base is not None:
        array = array.base
    return array


class NamedArrayTests(unittest.TestCase):

    def test_constructor_has_empty_names(self):
        n = NamedArray((2, 3))
        self.assertIsInstance(n, NamedArray)
        self.assertEqual(n.rownames, [])
        self.assertEqual(n.colnames, [])
        self.assertGreaterEqual(2, sys.getrefcount(n.rownames))
        self.assertGreaterEqual(2, sys.getrefcount(n.colnames))

    def test_view_cast_from_ndarray(self):
        n = np.zeros((2, 3)).view(NamedArray)
        self.assertIsInstance(n, NamedArray)
        self.assertEqual(n.rownames, [])

    def test_view_cast_to_ndarray_and_masked_array(self):
        n = NamedArray((2, 3))
        self.assertIsInstance(n.view(np.ndarray), np.ndarray)
        self.assertIsInstance(n.view(np.ma.MaskedArray), np.ma.MaskedArray)

    def test_slice_is_named_array_with_empty_names(self):
        n = NamedArray((2, 3))
        n.rownames = ["R1", "R2"]
        v = n[1:, :]
        self.assertIsInstance(v, NamedArray)
        self.assertEqual((1, 3), v.shape)
        self.assertEqual(v.rownames, [])

    def test_assign_names(self):
        n = NamedArray((2, 3))
        n.rownames = ["R1", "R2"]
        n.colnames = ["C1", "C2", "C3"]
        self.assertEqual(n.rownames, ["R1", "R2"])
        self.assertEqual(n.colnames, ["C1", "C2", "C3"])

    def test_equality(self):
        n = np.zeros((2, 3)).view(NamedArray)
        n2 = np.zeros((2, 3)).view(NamedArray)
        self.assertTrue((n == n2).all())

    def test_index_by_column_name(self):
        n = np.arange(6.0).reshape(2, 3).view(NamedArray)
        n.colnames = ["a", "b", "c"]
        col = n["b"]
        self.assertEqual(type(col), np.ndarray)
        np.testing.assert_array_equal(col, [1.0, 4.0])
        col[0] = 99
        self.assertEqual(n[0, 1], 1.0)

    def test_index_by_row_name(self):
        n = np.arange(6.0).reshape(2, 3).view(NamedArray)
        n.rownames = ["x", "y"]
        row = n["y"]
        self.assertEqual(type(row), np.ndarray)
        np.testing.assert_array_equal(row, [3.0, 4.0, 5.0])

    def test_column_name_wins_over_row_name(self):
        n = np.arange(4.0).reshape(2, 2).view(NamedArray)
        n.rownames = ["a", "b"]
        n.colnames = ["a", "c"]
        np.testing.assert_array_equal(n["a"], [0.0, 2.0])

    def test_unknown_name_raises_index_error(self):
        n = NamedArray((2, 3))
        n.colnames = ["a", "b", "c"]
        with self.assertRaises(IndexError):
            n["nope"]

    def test_ordinary_indexing_still_works(self):
        n = np.arange(6.0).reshape(2, 3).view(NamedArray)
        self.assertEqual(n[1, 2], 5.0)
        n[0, 0] = 7
        self.assertEqual(n[(0, 0)], 7.0)

    def test_str_with_names(self):
        n = np.array([[1.0, 2.5], [3.0, 4.0]]).view(NamedArray)
        n.rownames = ["r1", "r2"]
        n.colnames = ["time", "x"]
        expected = (
            "      time,   x\n"
            "r1 [[    1, 2.5],\n"
            "r2  [    3,   4]]\n"
        )
        self.assertEqual(str(n), expected)
        self.assertEqual(repr(n), expected)

    def test_str_without_names(self):
        n = np.array([[1.0, 2.0]]).view(NamedArray)
        self.assertEqual(str(n), " [[ 1, 2]]\n")

    def test_str_falls_back_for_1d_and_empty(self):
        self.assertEqual(str(np.arange(3.0).view(NamedArray)), str(np.arange(3.0)))
        self.assertEqual(str(NamedArray((0, 3))), str(np.empty((0, 3))))

    def test_getstate_contents(self):
        n = NamedArray((2, 3))
        n.rownames = ["R1", "R2"]
        n.colnames = ["C1", "C2", "C3"]
        state = n.__getstate__()
        self.assertEqual(
            sorted(state.keys()),
            ["_pickle_version", "array", "colnames", "dim1", "dim2", "nDims", "rownames"],
        )
        self.assertIsInstance(state["array"], bytes)
        self.assertEqual(len(state["array"]), 6 * 8)
        self.assertEqual(state["nDims"], 2)
        self.assertEqual((state["dim1"], state["dim2"]), (2, 3))
        self.assertEqual(state["rownames"], ["R1", "R2"])
        self.assertEqual(state["colnames"], ["C1", "C2", "C3"])
        self.assertEqual(state["_pickle_version"], 5)

    def test_getstate_1d(self):
        state = np.arange(4.0).view(NamedArray).__getstate__()
        self.assertEqual((state["nDims"], state["dim1"], state["dim2"]), (1, 4, 0))

    def test_getstate_rejects_3d(self):
        with self.assertRaises(ValueError):
            NamedArray((2, 2, 2)).__getstate__()

    def test_reduce_ex(self):
        n = NamedArray((2, 3))
        c, args, state, listitems, dictitems = n.__reduce_ex__(5)
        self.assertTrue(callable(c))
        self.assertEqual(args, ((2, 3),))
        self.assertIsNone(listitems)
        self.assertIsNone(dictitems)
        rebuilt = c(*args)
        self.assertIsInstance(rebuilt, NamedArray)
        rebuilt.__setstate__(state)
        self.assertEqual(rebuilt.shape, (2, 3))

    def test_getstate_and_setstate(self):
        n = NamedArray((2, 3))
        n[(0, 0)] = 9
        other = NamedArray(n.shape)
        other.__setstate__(n.__getstate__())
        self.assertAlmostEqual(other[(0, 0)], 9)

    def test_setstate_rejects_bad_input(self):
        n = NamedArray((2, 3))
        state = n.__getstate__()
        with self.assertRaises(ValueError):
            n.__setstate__([])
        with self.assertRaises(KeyError):
            n.__setstate__({k: v for k, v in state.items() if k != "_pickle_version"})
        with self.assertRaises(ValueError):
            n.__setstate__({**state, "_pickle_version": 4})
        with self.assertRaises(ValueError):
            NamedArray((3, 3)).__setstate__(state)
        with self.assertRaises(TypeError):
            n.__setstate__({**state, "array": "text"})

    def test_pickle_round_trip(self):
        n = np.arange(6.0).reshape(2, 3).view(NamedArray)
        n.rownames = ["R1", "R2"]
        n.colnames = ["C1", "C2", "C3"]
        for protocol in range(2, pickle.HIGHEST_PROTOCOL + 1):
            loaded = pickle.loads(pickle.dumps(n, protocol))
            self.assertIsInstance(loaded, NamedArray)
            self.assertEqual(loaded.rownames, ["R1", "R2"])
            self.assertEqual(loaded.colnames, ["C1", "C2", "C3"])
            self.assertTrue((loaded == n).all())

    def test_pickle_round_trip_without_names(self):
        n = np.arange(6.0).reshape(2, 3).view(NamedArray)
        loaded = pickle.loads(pickle.dumps(n))
        self.assertEqual(loaded.rownames, [])
        self.assertTrue((loaded == n).all())

    def test_pickle_round_trip_1d(self):
        n = np.arange(4.0).view(NamedArray)
        loaded = pickle.loads(pickle.dumps(n))
        self.assertEqual(loaded.shape, (4,))
        self.assertTrue((loaded == n).all())


class MatrixToArrayTests(unittest.TestCase):

    def setUp(self):
        self.source = np.arange(6.0).reshape(2, 3).copy()
        self.address = self.source.ctypes.data

    def test_zero_copy_shares_memory(self):
        out = matrix_to_array(self.address, 2, 3, ["r1", "r2"], ["a", "b", "c"], False, True)
        self.assertIsInstance(out, NamedArray)
        self.assertEqual(out.shape, (2, 3))
        self.assertEqual(out.rownames, ["r1", "r2"])
        self.assertEqual(out.colnames, ["a", "b", "c"])
        self.assertTrue(np.shares_memory(out, self.source))
        out[0, 0] = 42
        self.assertEqual(self.source[0, 0], 42)

    def test_zero_copy_is_writeable(self):
        out = matrix_to_array(self.address, 2, 3, [], [], False, False)
        self.assertTrue(out.flags.writeable)
        self.assertTrue(out.flags.c_contiguous)

    def test_zero_copy_keeps_owner_alive(self):
        class Owner:
            pass

        owner = Owner()
        out = matrix_to_array(self.address, 2, 3, [], ["a", "b", "c"], False, True, owner=owner)
        self.assertIs(_root_base(out)._owner, owner)
        # slices keep the chain too
        self.assertIs(_root_base(out[:, 1:])._owner, owner)

    def test_copy_is_independent(self):
        out = matrix_to_array(self.address, 2, 3, ["r1", "r2"], ["a", "b", "c"], True, True)
        self.assertIsInstance(out, NamedArray)
        self.assertFalse(np.shares_memory(out, self.source))
        np.testing.assert_array_equal(out, self.source)
        out[0, 0] = 42
        self.assertEqual(self.source[0, 0], 0)

    def test_single_column_without_names_is_1d_ndarray(self):
        column = np.array([1.0, 2.0, 3.0])
        for copy in (True, False):
            out = matrix_to_array(column.ctypes.data, 3, 1, [], [], copy, True)
            self.assertEqual(type(out), np.ndarray)
            self.assertEqual(out.shape, (3,))
            np.testing.assert_array_equal(out, column)

    def test_single_column_with_names_stays_2d_named(self):
        column = np.array([1.0, 2.0, 3.0])
        out = matrix_to_array(column.ctypes.data, 3, 1, [], ["x"], False, True)
        self.assertIsInstance(out, NamedArray)
        self.assertEqual(out.shape, (3, 1))

    def test_named_false_gives_plain_2d(self):
        out = matrix_to_array(self.address, 2, 3, ["r1", "r2"], ["a", "b", "c"], False, False)
        self.assertEqual(type(out), np.ndarray)
        self.assertEqual(out.shape, (2, 3))

    def test_empty_matrix_does_not_touch_memory(self):
        out = matrix_to_array(0, 0, 3, [], ["a", "b", "c"], False, True)
        self.assertEqual(out.shape, (0, 3))
        self.assertIsInstance(out, NamedArray)

    def test_structured(self):
        out = matrix_to_array(self.address, 2, 3, [], ["a", "b", "c"], True, True, structured=True)
        self.assertEqual(out.shape, (2,))
        self.assertEqual(out.dtype.names, ("a", "b", "c"))
        np.testing.assert_array_equal(out["b"], [1.0, 4.0])
        self.assertFalse(np.shares_memory(out, self.source))

    def test_structured_edge_cases(self):
        self.assertIsNone(matrix_to_array(0, 2, 0, [], [], True, True, structured=True))
        with self.assertRaises(ValueError):
            matrix_to_array(self.address, 2, 3, [], ["a"], True, True, structured=True)
        empty = matrix_to_array(0, 0, 2, [], ["a", "b"], True, True, structured=True)
        self.assertEqual(empty.shape, (0,))


if __name__ == "__main__":
    unittest.main()
