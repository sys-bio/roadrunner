# Lazy diagnostics in `Setting::getAs`

Formatting conversion errors only when a conversion fails reduced thread CPU time for a 101-point `RoadRunner.simulate` call by 13% on HCTZ and 19% on midazolam. At 1,001 points the reductions were 43% and 45%. CVODE's integration loop reads settings such as `multiple_steps` and `variable_step_size` on each call to advance toward an output time. The perfusion ratio was close to the difference observed with an identical binary.

## Public Python simulation

Each time is in milliseconds of thread CPU time for `runner.simulate(0, end, points)`, including construction of the returned `NamedArray`. Speedup is baseline time divided by candidate time within a block. The range spans the three block ratios. Baseline and candidate times are summarized independently, so their quotient can differ from the median paired speedup.

| Model | Points | Baseline (ms) | Candidate (ms) | Paired speedup | Block range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Demo | 101 | 0.129 | 0.058 | 2.28× | 2.24–2.35 |
| Repressilator | 101 | 0.310 | 0.227 | 1.37× | 1.37–1.39 |
| HCTZ | 101 | 0.573 | 0.497 | 1.15× | 1.13–1.16 |
| Midazolam | 101 | 0.429 | 0.348 | 1.23× | 1.21–1.23 |
| Repeated dose | 101 | 2.145 | 1.427 | 1.47× | 1.13–1.50 |
| Liver perfusion | 101 | 46.254 | 44.985 | 1.02× | 1.01–1.23 |
| Demo | 1,001 | 1.025 | 0.330 | 3.15× | 3.10–3.27 |
| Repressilator | 1,001 | 1.238 | 0.522 | 2.37× | 2.30–2.38 |
| HCTZ | 1,001 | 1.719 | 0.976 | 1.77× | 1.73–1.78 |
| Midazolam | 1,001 | 1.702 | 0.944 | 1.81× | 1.80–1.83 |
| Repeated dose | 1,001 | 3.728 | 1.846 | 1.94× | 1.51–2.04 |

Each block used four fresh Python processes in baseline, candidate, candidate, baseline order (12 processes total). Each case had three warmups. The first baseline process calibrated the repetition count to at least four runs and approximately 30 ms per case. All processes reused those counts. For each block we took the geometric mean of the two process medians for each build, then the ratio. The reported spread describes those three blocks. No confidence interval was estimated.

A second run assigned the **same baseline binary** to both labels, with two blocks and the original repetition counts. It showed apparent differences of up to about 2% at 101 points. This limits what can be inferred from the 1.02× perfusion result.

| Control, 101 points | Median A/B ratio | Block range |
| --- | ---: | ---: |
| Demo | 1.0179 | 1.0176–1.0183 |
| HCTZ | 0.9980 | 0.9889–1.0070 |
| Liver perfusion | 1.0142 | 1.0135–1.0148 |

An isolated C++ measurement timed 300,000 successful conversions per round. Across five rounds the median fell from 278.3 to 15.3 ns for a stored bool read as bool, and from 266.3 to 13.1 ns for a stored double read as double. The simulation table measures how much of that saving reaches a caller. [Conversion microbenchmark](microbenchmark.cpp)

## Workloads and verification

All runs used CPU 2 of an Intel Core i7-12700, GCC 13.3.0 Release (`-O3 -DNDEBUG`, C++20), LLVM 23.1.4 LLJIT at O2, SUNDIALS 7.9.0 with 64-bit indices, SWIG 4.5.1, Python 3.14.10 and NumPy 2.5.4. The solver was CVODE BDF with KLU, a colored sparse difference-quotient Jacobian, relative tolerance 1e-6, base absolute tolerance 1e-10, a 20,000-step limit and fixed output times. Solver timing instrumentation was disabled.

Each trajectory starts at time zero and includes time followed by every state-vector ID in model order. Demo and repressilator ran to 100 (7 columns each), HCTZ and midazolam to 1,440 (29 and 39 columns), repeated dose to 168 (5 columns), and perfusion to 60 (920 columns). HCTZ used `init(PODOSE_hctz) = 25` and midazolam used `init(PODOSE_mid) = 10`. The other models used their SBML initial conditions.

The simulation clock starts after model compilation and `resetAll()`. The archives include model setup, separate reset and simulation samples, reset-inclusive summaries, wall and thread CPU time, model hashes, ordered selection-ID hashes/counts and build fingerprints. They also include timing of the lower-level `_simulate` entry point with prepared options. Across the 22 cases and 14,136 trajectories (13,344 timed and 792 warmups), both builds and both entry points agreed on every float64 output byte, column names, shape and all recorded solver counters. [Simulation measurements](raw_results.json.gz) · [Same-binary controls](control_results.json.gz)

A separate conversion check compared 25,848 results for explicit and implicit conversions. Return values and all 7,456 thrown exceptions matched, including exception types and messages. The candidate also passed the check under UndefinedBehaviorSanitizer. The comparison includes 1,428 implementation-defined integer results observed on this platform. It excludes 5,480 probes of undefined or platform-sensitive floating-point conversions. [Conversion results](conversion_results.json) · [Conversion checker](check_conversions.cpp)

## Reproduction

The recorded binaries were built from `291d2932161f`, with only `source/Setting.h` changed between builds. This PR applies the conversion change to `f35126ec5`. Both revisions have identical native sources. The intervening Python typing change adds annotation metadata to `NamedArray` and preserves its runtime logic. The reported numbers come from that earlier A/B run. [Build and archive fingerprints](provenance.json)

Build the `roadrunner_python` target from the baseline and candidate with the same toolchain and dependencies. Use separate Release build directories with `-DBUILD_PYTHON=ON -DRR_USE_KLU=ON`. Each build produces an importable `lib/site-packages` directory. Copy the example model manifest, replace its paths with the prepared XML files and retain the recorded SHA-256 values. The first five inputs follow the sbmlsim-core preparation recipe. The repeated-dose input is already in SBML form. [Model manifest](models.example.json) · [Preparation recipe](https://github.com/matthiaskoenig/sbmlsim-core/blob/e5a2273e0d03755d67c4df5b00c5dc9130d02292/bench/sbmlsim.toml) · [Repeated-dose model](https://github.com/matthiaskoenig/sbmlsim-core/blob/e5a2273e0d03755d67c4df5b00c5dc9130d02292/bench/models/repeated_dose.xml)

From this directory, run on Linux with a Python interpreter containing NumPy. Select an available CPU on the test machine.

```bash
python benchmark.py --baseline-site-packages /path/to/baseline/lib/site-packages \
    --candidate-site-packages /path/to/candidate/lib/site-packages \
    --python /path/to/python --model-config models.json --cpu 2 --blocks 3 \
    --output run.json.gz
python benchmark.py --baseline-site-packages /path/to/baseline/lib/site-packages \
    --same-binary-control --models demo hctz perfusion --counts-from run.json.gz \
    --python /path/to/python --model-config models.json --cpu 2 --blocks 2 \
    --output control.json.gz
python benchmark.py --summarize run.json.gz --operation simulate --clock thread_cpu_ns
```

The published archives replace local absolute paths with artifact-relative labels and
ordered selection-ID lists with hashes and counts. Timing samples, trajectory hashes,
solver counters and summaries are unchanged; `provenance.json` records the original
JSON hashes. Model bytes are not redistributed.

To rerun the conversion comparison against the baseline header, from this directory:

```bash
python verify_conversions.py --baseline-ref f35126ec5 --sanitize
```

This compiles the standalone probe twice using the actual baseline and candidate
headers, compares returned value bits and complete exception messages, and repeats
the candidate's defined-case probe with UBSan and float-cast-overflow checks. It does
not run the optional undefined or platform-sensitive floating-conversion probes.

To repeat the conversion microbenchmark, compile `microbenchmark.cpp` with the
baseline and candidate `source` directories on the include path, using the same
compiler and `-O3 -DNDEBUG -std=c++20`. Run each binary on the selected CPU. No
`libroadrunner` build is needed for that isolated template measurement.
