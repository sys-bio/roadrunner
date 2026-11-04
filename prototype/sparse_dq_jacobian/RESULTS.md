# Sparse difference quotient Jacobian with KLU

This prototype answers one question that the Enzyme Jacobian prototype (#1418) left open under
"Not measured":

> A sparse difference quotient with the same coloring would also feed KLU, without Enzyme: one
> model evaluation per color, plus the finite difference error. It might be the cheaper route to
> most of the KLU speedup, if exactness turns out not to matter.

**Answer: on the four benchmark models, it is the whole KLU speedup, not most of it.** KLU with
the sparse difference quotient simulates as fast as KLU with the Enzyme Jacobian (Chen_MSB2009:
260 ms against 267 ms, 4.7x faster than dense with CVODE's difference quotient). It takes the
same number of steps, error test failures and Jacobians, and the results have the same error
against a reference. The exactness of the Jacobian does not matter here. A model loads as fast and
with as much memory as without the prototype, while the Enzyme Jacobian makes Chen_MSB2009 load in
3.8 s instead of 1.1 s, with a peak of 208 MiB instead of 120 MiB, and the build needs no Enzyme.
KLU also saves 3 MiB of the dense matrices on Chen_MSB2009.

## What it is

It reuses the parts of #1418 that do not need Enzyme:

- the conservative sparsity pattern of the Jacobian from the SBML math and the greedy column
  coloring (`JacobianSparsity`, unchanged except for the fallback checks moved into
  `JacobianSparsity::analyze`);
- the assembly of the Jacobian from `dv/dy` and the rate rule rows, `J = cf * N * dv/dy`;
- the CVODE settings `jacobian` and `linear_solver`, and the KLU build option `RR_USE_KLU`.

The Enzyme tangents are replaced by forward differences:
`LLVMExecutableModel::getStateVectorJacobianDQ` evaluates the reaction rates and the rate rule
rates at `y`, and once per color at `y` plus the increments of all columns of that color. Since the
columns of a color share no row of `[dv/dy; d(rate rule rates)/dy]`, each entry is the same forward
difference that a dense difference quotient computes. The CVODE callback takes the increments of
CVODE's own dense difference quotient (`cvLsDenseDQJac`: `max(sqrt(eps) |y_j|, minInc / ewt_j)`),
so the sparse difference quotient is CVODE's difference quotient, computed with fewer model
evaluations and without the cancellation in the sums of the reaction rates.

```python
r = roadrunner.RoadRunner("model.xml")
r.integrator.jacobian = "sparse_difference_quotient"
r.integrator.linear_solver = "klu"     # needs RR_USE_KLU
r.simulate(0, 100, 101)
r.integrator.getStatistics()           # #1425, plus num_jac_colors, num_jac_model_evals, sparse_linear_solver
```

Without a load option: the pattern and the coloring are computed for every model at load (0.2 ms
for hctz_body, 3.8 ms for Chen_MSB2009, against a load of 1.1 s), and the integrator settings alone
switch the sparse difference quotient and KLU on. Every model falls back to CVODE's dense
difference quotient when it has no pattern; the reasons are those of #1418 that do not concern
Enzyme: conserved moiety analysis, a stoichiometry set by a rule, a species conversion factor and
a global conversion factor that depends on the state. Functions that Enzyme cannot differentiate
(`factorial`, `rateOf`, the `distrib` functions, ...) are no problem for differences.

The branch is based on #1425 (the counters of the integrators), which the benchmark needs.

## Results

The four models and simulations of #1418: `simulate(0, end, 201)` at `rtol = atol = 1e-10`, LLJit
at the optimization level `DEFAULT`. The Enzyme numbers come from the branch of #1418, built with
`enzyme-counter-totals.patch`, so that its counters are summed over the reinitializations of CVODE
like those of #1425 (the counters in #1418 are those of the last reinitialization only; Chen_MSB2009
reinitializes CVODE 4 times, at piecewise switches). Every number is the median over 5 rounds, in
which the two builds and the models take turns, each in a process of its own; a simulation time is
the median of 10 runs, in which the configurations take turns.

- **speedup**: against dense + DQ of the same build.
- **model evaluations for Jacobians**: of the state vector rate for CVODE's difference quotient,
  of the reaction rates for the sparse one (colors + 1 per Jacobian).
- **one Jacobian**: for CVODE's difference quotient, n times one evaluation of the state vector
  rate; for the sparse one, the time in the Jacobian callback, measured with `collect_timing`;
  for Enzyme, the timing tool of #1418 at the middle of the timecourse.
- **error / tol**: the largest `|x - x_ref| / (atol + rtol |x_ref|)` over the output, against
  dense + DQ at `rtol = 1e-12`, `atol = 1e-14`.

| model | states | colors | configuration | simulate (ms) | speedup | steps | error test fails | Jacobians | model evaluations for Jacobians | one Jacobian (µs) | error / tol |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hctz_body | 28 | 5 | dense + DQ | 1.5 | 1.00x | 878 | 2 | 15 | 420 | 8.74 | 29.7 |
|  |  |  | dense + sparse DQ | 1.4 | 1.07x | 878 | 2 | 15 | 90 | 1.81 | 29.7 |
|  |  |  | KLU + sparse DQ | 1.21 | 1.24x | 878 | 2 | 15 | 90 | 1.81 | 29.7 |
|  |  |  | dense + DQ (Enzyme build) | 1.45 | 1.00x | 878 | 2 | 15 | 420 | 7.79 | 29.7 |
|  |  |  | KLU + Enzyme | 1.21 | 1.20x | 878 | 2 | 15 | 0 | 3.96 | 29.8 |
| glimepiride_body | 51 | 2 | dense + DQ | 1.1 | 1.00x | 338 | 0 | 6 | 306 | 10.1 | 38.4 |
|  |  |  | dense + sparse DQ | 1.05 | 1.05x | 338 | 0 | 6 | 18 | 0.963 | 38.4 |
|  |  |  | KLU + sparse DQ | 0.783 | 1.41x | 338 | 0 | 6 | 18 | 0.963 | 38.4 |
|  |  |  | dense + DQ (Enzyme build) | 1.03 | 1.00x | 338 | 0 | 6 | 306 | 8.74 | 38.4 |
|  |  |  | KLU + Enzyme | 0.744 | 1.39x | 338 | 0 | 6 | 0 | 0.967 | 38.4 |
| Smith_BMCSystBiol2013 | 104 | 3 | dense + DQ | 14.6 | 1.00x | 2162 | 19 | 39 | 4056 | 51.3 | 1.36e+03 |
|  |  |  | dense + sparse DQ | 14.4 | 1.01x | 2322 | 33 | 41 | 164 | 3.68 | 1.36e+03 |
|  |  |  | KLU + sparse DQ | 8.58 | 1.70x | 2162 | 19 | 39 | 156 | 3.68 | 1.36e+03 |
|  |  |  | dense + DQ (Enzyme build) | 14.4 | 1.00x | 2162 | 19 | 39 | 4056 | 49.2 | 1.36e+03 |
|  |  |  | KLU + Enzyme | 8.59 | 1.67x | 2162 | 19 | 39 | 0 | 2.85 | 1.36e+03 |
| Chen_MSB2009 | 500 | 7 | dense + DQ | 1.22e+03 | 1.00x | 3100 | 275 | 94 | 47000 | 1.85e+03 | 1.9e+05 |
|  |  |  | dense + sparse DQ | 1.06e+03 | 1.16x | 3013 | 278 | 95 | 760 | 83.8 | 1.1e+05 |
|  |  |  | KLU + sparse DQ | 260 | 4.70x | 1840 | 68 | 63 | 504 | 83.8 | 1.84e+05 |
|  |  |  | dense + DQ (Enzyme build) | 1.24e+03 | 1.00x | 3100 | 275 | 94 | 47000 | 1.8e+03 | 1.9e+05 |
|  |  |  | KLU + Enzyme | 267 | 4.63x | 1874 | 70 | 62 | 0 | 36 | 1.62e+05 |

| model | load (s) | load with the Enzyme Jacobian (s) |
| --- | ---: | ---: |
| hctz_body | 0.12 | 0.26 |
| glimepiride_body | 0.18 | 0.44 |
| Smith_BMCSystBiol2013 | 0.33 | 0.80 |
| Chen_MSB2009 | 1.14 | 3.80 |

What the numbers show:

- **KLU brings the speedup, and any Jacobian with the sparsity pattern makes it possible.** KLU +
  sparse DQ and KLU + Enzyme are within 6 % of each other on all four models, and both are 1.2 to
  4.7 times faster than dense + DQ. With the dense solver, the cheaper Jacobian gains at most 16 %,
  as the exact one did in #1418.
- **The steps do not depend on the exactness of the Jacobian.** KLU + sparse DQ and KLU + Enzyme
  take the same steps, error test failures and Jacobians on three models, and 2 % fewer steps
  on Chen_MSB2009 (1840 against 1874). Both take 40 % fewer steps than dense on Chen_MSB2009, with
  a quarter of the error test failures. So the difference that #1418 saw between KLU and dense
  comes from the linear solver, as #1418 supposed, and not from the exactness of the Jacobian. (#1418 reported 587 against 1349 steps; those were the steps after the last
  reinitialization only.)
- **The accuracy of the simulation is the same.** The error against the reference is the
  integration error at `1e-10`, the same with every Jacobian.
- **One sparse difference quotient costs about as much as one Enzyme Jacobian.** It is faster on
  hctz_body (1.8 µs against 4.0 µs), as fast on glimepiride_body, and slower on
  Smith_BMCSystBiol2013 (3.7 µs against 2.9 µs) and Chen_MSB2009 (84 µs against 36 µs, and 22
  times faster than CVODE's 1.85 ms). CVODE evaluates the Jacobian 63 times in the simulation of
  Chen_MSB2009, so the difference is 3 ms of 260 ms.
- **Few colors.** 2 to 7 colors for 28 to 500 states, so a Jacobian costs 3 to 8 evaluations of
  the reaction rates instead of 28 to 500 evaluations of the state vector rate.

## Memory

`memory.py` measures every model, build and configuration in a fresh process, as the median of 5
processes:

- **RSS after the load**: the resident set size after loading the model minus the RSS before; it
  holds the compiled code, which LLVM does not allocate with malloc.
- **peak RSS of the load**: the peak RSS during the load (code generation, the Enzyme pass and
  compilation) minus the RSS before.
- **heap after the simulation**: the memory allocated with malloc and in use after
  `simulate(0, end, 201)` (glibc `mallinfo2`), which holds the CVODE memory, the matrices and
  factorizations of the linear solver and the workspace of the Jacobian. Unlike the RSS, it drops
  when the dense matrices are replaced by the sparse ones, so it shows the difference between the
  linear solvers.
- **heap, change by the simulation**: the part of it allocated by the simulation: the copy of the
  Jacobian that CVODE keeps (`savedJ`), the KLU factorization, the workspace of the sparse
  difference quotient and the result.

| model | build and configuration | RSS after the load (MiB) | peak RSS of the load (MiB) | heap after the simulation (MiB) | heap, change by the simulation (MiB) |
| --- | --- | ---: | ---: | ---: | ---: |
| hctz_body | dense + DQ | 30.9 | 30.9 | 22.32 | 0.05 |
|  | dense + sparse DQ | 30.9 | 30.9 | 22.33 | 0.06 |
|  | KLU + sparse DQ | 30.9 | 30.9 | 22.32 | 0.06 |
|  | #1418 build without Enzyme: dense + DQ | 32.4 | 32.4 | 22.40 | 0.05 |
|  | #1418 build with Enzyme: dense + DQ | 39.3 | 39.3 | 22.46 | 0.05 |
|  | #1418 build with Enzyme: KLU + Enzyme | 39.3 | 39.3 | 22.45 | 0.06 |
| glimepiride_body | dense + DQ | 35.1 | 35.1 | 23.72 | 0.11 |
|  | dense + sparse DQ | 35.1 | 35.1 | 23.73 | 0.11 |
|  | KLU + sparse DQ | 35.1 | 35.1 | 23.70 | 0.10 |
|  | #1418 build without Enzyme: dense + DQ | 36.5 | 36.5 | 23.81 | 0.11 |
|  | #1418 build with Enzyme: dense + DQ | 47.7 | 47.7 | 23.85 | 0.11 |
|  | #1418 build with Enzyme: KLU + Enzyme | 47.7 | 47.7 | 23.82 | 0.10 |
| Smith_BMCSystBiol2013 | dense + DQ | 51.9 | 51.9 | 32.97 | 0.29 |
|  | dense + sparse DQ | 51.9 | 51.9 | 33.00 | 0.32 |
|  | KLU + sparse DQ | 51.9 | 51.9 | 32.90 | 0.29 |
|  | #1418 build without Enzyme: dense + DQ | 53.4 | 53.4 | 33.04 | 0.29 |
|  | #1418 build with Enzyme: dense + DQ | 64.7 | 64.7 | 33.11 | 0.29 |
|  | #1418 build with Enzyme: KLU + Enzyme | 64.6 | 64.6 | 33.05 | 0.30 |
| Chen_MSB2009 | dense + DQ | 110.3 | 119.8 | 55.98 | 2.68 |
|  | dense + sparse DQ | 110.3 | 119.8 | 56.14 | 2.81 |
|  | KLU + sparse DQ | 110.3 | 119.8 | 53.02 | 1.53 |
|  | #1418 build without Enzyme: dense + DQ | 111.2 | 120.4 | 56.03 | 2.68 |
|  | #1418 build with Enzyme: dense + DQ | 210.7 | 210.7 | 56.32 | 2.68 |
|  | #1418 build with Enzyme: KLU + Enzyme | 181.7 | 208.3 | 53.36 | 1.54 |

What the numbers show:

- **The sparse difference quotient costs no memory worth mentioning.** The pattern, the coloring
  and the workspace of the difference quotient add 0.16 MiB to the heap on Chen_MSB2009 and less
  than 0.04 MiB on the other models. The RSS after the load is the same as without them (and 0.9 to
  1.5 MiB below the #1418 build without Enzyme, which is based on an older `develop`).
- **KLU saves memory on the large model.** The dense solver keeps two dense n x n matrices (the
  Jacobian and its saved copy, 2 x 1.9 MiB for 500 states); KLU keeps them sparse (4400 nonzeros)
  plus its factorization. The heap after the simulation of Chen_MSB2009 is 3.0 MiB smaller with
  KLU, with either Jacobian. On the small models the difference is below 0.1 MiB.
- **The Enzyme Jacobian costs memory at load.** The derivative code adds 7 to 11 MiB of RSS on
  the three small models and 70 to 100 MiB on Chen_MSB2009, where the RSS after the load varies
  between processes (182 to 211 MiB) and the peak of the load is 208 to 211 MiB, against 120 MiB
  without Enzyme.

## Validation

- **Against the dense difference quotient.** `SparseDQJacobianTests.SbmlTestSuite` compares the
  sparse difference quotient with the dense forward difference of the state vector rate with the
  same increments, at the start and at t = 2, on every semantic case of the SBML test suite that
  roadrunner loads, with MCJit and with LLJit. They agree to rounding (100 eps times the size of
  the rates divided by the increment) in all 1465 cases with a pattern, and the dense difference
  has no nonzero outside the pattern. 150 cases fall back: 104 with a stoichiometry set by a rule
  and 46 with species conversion factors. 208 cases do not load. The test runs when
  `RR_SBML_TEST_SUITE` points to `test/sbml-test-suite/semantic`.
- **Simulations.** With the dense solver and with KLU, the sparse difference quotient gives the
  result of CVODE's dense difference quotient on small models with assignment rules, function
  definitions, piecewise, variable compartments, rate rules reading reaction rates, `factorial`
  and events (`SparseDQJacobianTests.CvodeSimulatesTheSame`), and on the benchmark models (table).
- **Tests.** `SparseDQJacobianTests` (6 tests, both JITs) and `JacobianSparsityTests` (8 tests,
  from #1418).

## Not covered

- The fallbacks: a stoichiometry set by a rule, species conversion factors and a state-dependent
  global conversion factor could be handled by differencing the state vector rate itself, with a
  coloring of the pattern of `J` instead of `dv/dy` (more colors, no assembly). Conserved moiety
  analysis needs the Jacobian with respect to the independent species.
- Models restored with `loadState` have no pattern and fall back.
- `getFullJacobian`, the steady-state solvers and forward sensitivities could use the sparse
  difference quotient too.
- KLU still needs SuiteSparse, and SUNDIALS built with it, in libroadrunner-deps and the wheels.
- Only measured on Linux (Ubuntu 24.04, gcc 13.3), on a machine shared with other jobs.

## How to reproduce

- Build with `-DRR_USE_KLU=ON -DRR_SUITESPARSE_PREFIX=<SuiteSparse install>` and a SUNDIALS built with
  KLU, as in `prototype/enzyme_jacobian/BUILD.md` of #1418 (without the Enzyme steps).
- `benchmark.py` writes `benchmark.csv` and `memory.py` writes `memory.csv`; both need this build
  and a build of the #1418 branch with `enzyme-counter-totals.patch`. `table.py` prints the
  tables above from them.
- `models.py` defines the four models, as in #1418.
