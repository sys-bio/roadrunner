# Checking the Python bindings

The `.pyi` files in `wrappers/Python/roadrunner` describe the public Python
API. The wheel ships them alongside `py.typed` so type checkers can discover
them after installation. When changing a SWIG binding, check its typemap in
`roadrunner.i` for the Python argument and result types. For example,
`simulate` returns a `NamedArray` with column names, and model-specific
attributes such as `runner.S1` resolve to scalar values. Editing a stub
changes type checking and editor completions without changing how the binding
runs.

Native APIs without container conversions expose opaque SWIG handles, which
the stubs distinguish from Python lists and NumPy arrays. Matrix methods use
the default named-array configuration; disabling named matrices or enabling
structured simulation output changes the runtime array representation. A
nonempty `simulate(output_file=...)` returns a status string instead of an array.
The local solver override exceptions preserve the argument differences in the
existing bindings. The annotation checks reject unused exceptions.

Run the following from the repository root. The fixtures check the types
inferred for valid calls and the diagnostics produced for invalid calls.

```bash
uv venv .venv
source .venv/bin/activate
uv pip install numpy typing-extensions ty ruff pytest 'swig>=4.4.1'
ty check --python .venv
ruff check
ruff format --check
python -m pytest -q test/python/test_named_array_module.py
```

Regenerate the Python proxy with SWIG and check its public names against the
stub. SWIG only needs the version header's declarations for this check.

```bash
bindings=wrappers/Python/roadrunner
proxy_dir=$(mktemp -d)
cp source/rrVersionInfo.h.in "$proxy_dir/rrVersionInfo.h"
swig -c++ -python -threads -doxygen -keyword -w511 \
    -I"$proxy_dir" -Isource -Iwrappers/Python/include -I"$bindings" \
    -o "$proxy_dir/roadrunner_wrap.cxx" "$bindings/roadrunner.i"
python test/typing/check_swig_coverage.py "$proxy_dir/roadrunner.py" \
    --package-stub "$bindings/__init__.pyi"
```
