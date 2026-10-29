"""NamedArray: a numpy ndarray that carries row and column names.

Also holds the factory that turns matrix memory handed over by the
compiled extension into numpy arrays without using the numpy C API.
"""
import numpy as np

PICKLE_VERSION = 5
PICKLE_VERSION_KEY = "_pickle_version"


def _format_table(rows_of_values, rownames, colnames):
    """Render a 2D list of floats as a table with optional row and column names."""
    rnames = [str(s) for s in rownames]
    cnames = [str(s) for s in colnames]
    cells = [[format(v, "g") for v in row] for row in rows_of_values]
    nrows = len(cells)
    ncols = len(cells[0])

    row_name_width = max((len(s) for s in rnames), default=0) + 1

    col_widths = []
    for j in range(ncols):
        width = len(cnames[j]) + 1 if j < len(cnames) else 0
        for i in range(nrows):
            width = max(width, len(cells[i][j]) + 1)
        col_widths.append(width)

    lines = []
    if cnames:
        parts = []
        for j in range(ncols):
            name = cnames[j] if j < len(cnames) else ""
            width = (row_name_width + 2 if j == 0 else 0) + col_widths[j]
            parts.append(name.rjust(width) + ("," if j < ncols - 1 else ""))
        lines.append("".join(parts))

    for i in range(nrows):
        name = rnames[i] if i < len(rnames) else ""
        opener = " [" if i > 0 else "[["
        body = ",".join(cells[i][j].rjust(col_widths[j]) for j in range(ncols))
        closer = "]," if i < nrows - 1 else "]]"
        lines.append(name.ljust(row_name_width) + opener + body + closer)

    return "".join(line + "\n" for line in lines)


class NamedArray(np.ndarray):
    """An ndarray with ``rownames`` and ``colnames`` lists.

    A 2D NamedArray can be indexed by a row or column name, which returns a
    1D copy: ``arr["time"]``.
    """

    # Pickles refer to the class by this module path.
    __module__ = "roadrunner._roadrunner"

    def __array_finalize__(self, obj):
        self.rownames = []
        self.colnames = []

    def __getitem__(self, key):
        if isinstance(key, str) and self.ndim == 2:
            plain = self.view(np.ndarray)
            if key in self.colnames:
                return plain[:, self.colnames.index(key)].copy()
            if key in self.rownames:
                return plain[self.rownames.index(key), :].copy()
        return super().__getitem__(key)

    def __str__(self):
        plain = self.view(np.ndarray)
        if plain.ndim != 2 or 0 in plain.shape or plain.dtype.kind != "f":
            return str(plain)
        return _format_table(plain.tolist(), self.rownames, self.colnames)

    __repr__ = __str__

    def __getstate__(self):
        if self.ndim not in (1, 2):
            raise ValueError(f"Unexpected number of dimensions {self.ndim}")
        return {
            "array": np.ascontiguousarray(self, dtype=np.float64).tobytes(),
            "nDims": self.ndim,
            "dim1": int(self.shape[0]),
            "dim2": int(self.shape[1]) if self.ndim == 2 else 0,
            "rownames": list(self.rownames),
            "colnames": list(self.colnames),
            PICKLE_VERSION_KEY: PICKLE_VERSION,
        }

    def __reduce_ex__(self, protocol):
        state = self.__getstate__()
        return (type(self), (self.shape,), state, None, None)

    def __setstate__(self, state):
        if not isinstance(state, dict):
            raise ValueError("__setstate__ input object is not a dict")
        if PICKLE_VERSION_KEY not in state:
            raise KeyError(f"No '{PICKLE_VERSION_KEY}' in pickled data")
        if state[PICKLE_VERSION_KEY] != PICKLE_VERSION:
            raise ValueError(
                f"Pickle version mismatch. Got version {state[PICKLE_VERSION_KEY]} "
                f"but expected version {PICKLE_VERSION}."
            )

        data = state["array"]
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(f"Cannot accept object of type {type(data).__name__}, expected a bytes object")

        n_dims = state["nDims"]
        if n_dims == 1:
            shape = (state["dim1"],)
        elif n_dims == 2:
            shape = (state["dim1"], state["dim2"])
        else:
            raise ValueError(f"Unexpected number of dimensions {n_dims}")

        values = np.frombuffer(data, dtype=np.float64)
        if tuple(self.shape) != shape or values.size != int(np.prod(shape)):
            raise ValueError(f"State of shape {shape} does not fit an array of shape {self.shape}")
        self[...] = values.reshape(shape)

        self.rownames = list(state["rownames"] or [])
        self.colnames = list(state["colnames"] or [])


class _Buffer:
    """Exposes raw double memory to numpy and keeps its owner alive."""

    def __init__(self, address, shape, owner):
        self.__array_interface__ = {
            "data": (address, False),
            "shape": tuple(shape),
            "typestr": np.dtype(np.float64).str,
            "version": 3,
        }
        self._owner = owner


def _wrap(address, shape, owner, copy):
    if 0 in shape:
        return np.empty(shape, dtype=np.float64)
    buffer = _Buffer(address, shape, owner)
    if copy:
        return np.array(buffer, dtype=np.float64, copy=True)
    return np.asarray(buffer)


def matrix_to_array(address, rows, cols, rownames, colnames, copy, named, structured=False, owner=None):
    """Build a numpy array from a row-major block of doubles at ``address``.

    With ``copy`` false the array shares the memory and holds a reference to
    ``owner``. A single-column matrix without column names becomes a 1D
    ndarray. Otherwise the result is a NamedArray when ``named`` is true and
    a 2D ndarray when it is false. ``structured`` returns a 1D structured
    array with one float field per column name (always a copy).
    """
    if structured:
        if cols == 0:
            return None
        if cols != len(colnames):
            raise ValueError("column names size does not match matrix columns size")
        dtype = np.dtype([(str(name), "f8") for name in colnames])
        if rows == 0:
            return np.empty(0, dtype=dtype)
        values = _wrap(address, (rows, cols), owner, True)
        return values.view(dtype).reshape(rows)

    if cols == 1 and len(colnames) == 0:
        return _wrap(address, (rows,), owner, copy)

    array = _wrap(address, (rows, cols), owner, copy)
    if not named:
        return array

    result = array.view(NamedArray)
    result.rownames = list(rownames)
    result.colnames = list(colnames)
    return result
