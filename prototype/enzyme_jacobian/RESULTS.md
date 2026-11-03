# Enzyme Jacobian and KLU for CVODE: results of the prototype

The question was whether the exact Jacobian computed by Enzyme on roadrunner's generated LLVM IR, together with the sparse KLU solver, is worth pursuing in roadrunner. Design: sys-bio/roadrunner #1412 option 3 combined with #1409. Branch `enzyme-jacobian` of the fork, build instructions in `BUILD.md`.

## Answer

Yes, with one large cost to solve first:

- **Exact.** The Enzyme Jacobian matches finite differences on all four benchmark models at three states each, and on all 1417 cases of the SBML test suite it covers. The sparsity pattern contained every finite difference nonzero.
- **Fast per Jacobian.** One Enzyme Jacobian costs 1.9 to 47 times less than CVODE's dense difference quotient, and the advantage grows with the size of the model.
- **Fast per simulation, but only together with KLU.** With the dense solver the Enzyme Jacobian gains only 4 to 14 %, because CVODE evaluates a Jacobian only 6 to 27 times per simulation. With KLU the simulations are 1.2 to 5.0 times faster, more so the larger the model (Chen_MSB2009, 500 states: 4.95x).
- **Slow to load.** Compiling the derivative code is expensive: loading takes 2 to 7 times longer, 38 s instead of 6 s for Chen_MSB2009 on the loaded machine (11 s instead of 2.2 s when measured alone). This is the main problem to solve before the feature is upstreamed.
- **No cost when switched off.** A build with Enzyme and KLU, with the feature switched off, loads and simulates within 3 % of `develop`.

## Go/no-go gate

Passed with the tagged Enzyme release `v0.0.303`, built against roadrunner's LLVM 23.1.4, with no patches. Two things were needed:

- **IR bodies for some helper functions.** The generated code calls C++ helpers (`rr_min`, `rr_max`, `rr_logd`, `rr_rootd`, `arccot`, `rr_arccot_negzero`) that Enzyme cannot see into. They get IR bodies in the module.
- **A diagnostic handler.** Enzyme reports failures as LLVM error diagnostics, which terminate the process by default. A diagnostic handler turns them into exceptions, so a model Enzyme cannot handle falls back to the difference quotient instead of crashing.

## Validation (`validation.md`)

- **Reference.** The Enzyme Jacobian is compared with central differences of the state vector rate, using a ladder of step sizes, within 1e-6 relative, plus one-sided differences within 1e-4 relative for kinks of `max` and `min`. The one-sided tolerance applies to every entry, so the effective tolerance is 1e-4: of the 1417 passing cases, one has a largest error above 1e-6 (01395, 5.2e-5, at a state of 2.8e-10, where two finite difference references already disagree by 1.6e-4); the next largest is 1.0e-6.
- **Benchmark models.** No mismatches and no finite difference nonzeros outside the sparsity pattern. The largest relative error is 5.5e-7.
- **SBML test suite.** Of 1823 semantic cases:
  - 1417 pass.
  - 164 fall back, with a reason: 104 have a stoichiometry set by a rule, 46 have species conversion factors, and 14 call a function that has no derivative (10 of them `csr_matrix_get_nz`, which `rateOf` calls to read a stoichiometry, 3 `factorial`, 1 `sec`).
  - 242 are skipped: roadrunner cannot simulate 208 of them (algebraic rules, delays, fast reactions), and 34 have no timecourse.
  - None fail.
- **Simulations.** All three configurations stay within the integration error of a tight-tolerance reference (`rtol = 1e-12`, `atol = 1e-14`). In test case 00197 the difference quotient is the less accurate one.
- **Bugs found and fixed:**
  - The model cache key ignored the Enzyme option.
  - The sparsity pattern missed the amount conversion of rate rules on concentration species (16 test suite cases).
  - The final code review found that loading or regenerating a model the `jacobian`/`linear_solver` settings could not serve (a fallback model, a load without the Enzyme option, conserved moiety analysis) made CVODE throw halfway through its setup, and the next simulation crashed. CVODE now falls back to the dense difference quotient with a warning on every such path, including copies of a RoadRunner.

## Benchmark (`benchmark.csv`)

All three configurations come from the same release build. Settings: `rtol = atol = 1e-10`, `simulate(0, end, 201)`, medians of 20 repetitions with the configurations taking turns.

- **Jacobian timings** are measured with `enzyme_jacobian_timing` at the state in the middle of the timecourse. The difference quotient is counted as n evaluations of the state vector rate, which is what CVODE's dense difference quotient costs.
- **Shared machine.** Other jobs were running at the same time (load average 6 to 18 on 20 cores). Absolute times vary up to 2x between runs; the ratios are stable within about 10 %.

`glimepiride_body` is `glimepiride_body_flat.xml` of `~/git/glimepiride-model/models` with 51 states, not the 168-state model of the #1409 profiles; `models.py` names every file and the PEtab commit.

| model | states | J nnz | colors | load (s) | load with Enzyme (s) | Jacobian, difference quotient (µs) | Jacobian, Enzyme (µs) | speedup | simulate, dense + DQ (ms) | dense + Enzyme (ms) | KLU + Enzyme (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hctz_body | 28 | 91 | 5 | 0.21 | 0.42 | 9.2 | 4.92 | 1.9x | 1.79 | 1.72 (1.04x) | 1.47 (1.21x) |
| glimepiride_body | 51 | 125 | 2 | 0.32 | 0.67 | 10.0 | 1.24 | 8.1x | 1.31 | 1.24 (1.06x) | 0.91 (1.44x) |
| Smith_BMCSystBiol2013 | 104 | 802 | 3 | 0.62 | 2.85 | 123.0 | 7.05 | 17.4x | 41.49 | 36.50 (1.14x) | 22.47 (1.85x) |
| Chen_MSB2009 | 500 | 4400 | 7 | 5.73 | 37.81 | 3919.6 | 83.96 | 46.7x | 1766 | 1602 (1.10x) | 357 (4.95x) |

CVODE counters of one simulation:

| model | configuration | steps | RHS evaluations | RHS evaluations for the Jacobian | Jacobian evaluations | Newton iterations | Newton failures | error test failures | linear solver setups |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hctz_body | dense + DQ | 878 | 923 | 420 | 15 | 922 | 0 | 2 | 77 |
| hctz_body | dense + Enzyme | 878 | 923 | 0 | 15 | 922 | 0 | 2 | 77 |
| hctz_body | KLU + Enzyme | 878 | 923 | 0 | 15 | 922 | 0 | 2 | 77 |
| glimepiride_body | dense + DQ | 338 | 359 | 306 | 6 | 356 | 0 | 0 | 35 |
| glimepiride_body | dense + Enzyme | 338 | 359 | 0 | 6 | 356 | 0 | 0 | 35 |
| glimepiride_body | KLU + Enzyme | 338 | 359 | 0 | 6 | 356 | 0 | 0 | 35 |
| Smith_BMCSystBiol2013 | dense + DQ | 1177 | 1265 | 2080 | 20 | 1262 | 0 | 9 | 89 |
| Smith_BMCSystBiol2013 | dense + Enzyme | 1199 | 1295 | 0 | 20 | 1292 | 0 | 6 | 85 |
| Smith_BMCSystBiol2013 | KLU + Enzyme | 1177 | 1265 | 0 | 20 | 1262 | 0 | 9 | 89 |
| Chen_MSB2009 | dense + DQ | 1349 | 2108 | 13000 | 26 | 2105 | 4 | 143 | 283 |
| Chen_MSB2009 | dense + Enzyme | 1429 | 2188 | 0 | 27 | 2185 | 8 | 157 | 310 |
| Chen_MSB2009 | KLU + Enzyme | 587 | 620 | 0 | 10 | 617 | 0 | 4 | 52 |

## Where the speedup comes from, and where it does not

- **Fewer model evaluations, which matters little.** The Enzyme Jacobian removes every right-hand-side evaluation of the difference quotient: 31 to 86 % of all evaluations. But CVODE reuses its Jacobian and evaluates it only 6 to 27 times per simulation. On these models evaluating the model is cheap next to the linear algebra, so the dense solver gains only 4 to 14 %.
- **Sparse linear algebra, which matters most.** KLU factorises and solves the sparse `I - γJ` instead of a dense matrix. The gain grows with the model, as the profiles of #1409 predicted: from 1.2x at 28 states to 5x at 500 states. KLU needs a Jacobian from roadrunner, so the Enzyme Jacobian is the precondition for it rather than the speedup itself.
- **More accurate Newton steps, which shows only on the largest model.** The exact Jacobian alone does not change the step counts: dense + Enzyme takes the same steps as dense + DQ on the PBPK models and slightly more on Smith and Chen. On Chen_MSB2009, KLU + Enzyme takes 587 steps instead of 1349, with 4 error test failures instead of 143. The dense solver with the exact Jacobian behaves like the dense difference quotient, so the gain comes from the linear solver. My hypothesis is that dense LU loses accuracy on this badly scaled matrix (values up to 1.2e9), and KLU, which scales the rows, does not. I have not verified this; it is worth a closer look, since it decides whether dense users would gain from a better-conditioned solve too.
- **The vector mode does not pay off.** The time of a Jacobian hardly depends on the vector width, while the compilation time grows with it. The width is therefore 1 by default, one seed direction per pass; the environment variable `RR_ENZYME_MAX_WIDTH` sets a larger one. Measured alone (`width.csv`, other jobs on the machine):

| model | colors | width | passes | load with Enzyme (s) | Enzyme Jacobian (µs) | difference quotient (µs) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hctz_body | 5 | 1 | 5 | 0.39 | 4.80 | 9.0 |
| hctz_body | 5 | 5 | 1 | 0.76 | 2.29 | 9.5 |
| Smith_BMCSystBiol2013 | 3 | 1 | 3 | 1.63 | 3.96 | 58 |
| Smith_BMCSystBiol2013 | 3 | 3 | 1 | 1.88 | 3.50 | 58 |
| Chen_MSB2009 | 7 | 1 | 7 | 11.4 | 47.3 | 2245 |
| Chen_MSB2009 | 7 | 2 | 4 | 13.3 | 42.6 | 2185 |
| Chen_MSB2009 | 7 | 4 | 2 | 27.8 | 41.3 | 2157 |
| Chen_MSB2009 | 7 | 7 | 1 | 58.7 | 43.6 | 2185 |

## Load time

Loading with the Enzyme Jacobian takes 2 to 7 times longer. Instrumenting the load of Chen_MSB2009 (`Enzyme Jacobian: preparation ... s, compilation ... s` at log level information) shows where the time goes:

- **Enzyme itself is not the problem.** The sparsity analysis and the Enzyme pass take 7 s at width 7.
- **Compiling the derivative code is.** MCJit takes 133 s for it in the same run (width 7, other jobs on the machine). Switching off roadrunner's IR optimisations makes compilation slower still, so the cost is in instruction selection and register allocation of very large straight-line functions, not in the optimisation passes.

Options for an upstream version:

- Put the derivative functions in a module of their own and compile it lazily, on the first Jacobian CVODE asks for, or in the background (LLJit can do this), so loading is not slowed down.
- Split the derivative per reaction or per block of reactions into smaller functions.
- Compile the derivative code at a lower optimisation and code generation level.

## Cost when switched off (`baseline.csv`)

`develop` (`2876ab2ed`) and the feature build with the feature off, both built against the same dependencies, run alternately twice. Medians:

| model | load, develop (s) | load, feature (s) | ratio | simulate, develop (ms) | simulate, feature (ms) | ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hctz_body | 0.208 | 0.209 | 1.00 | 1.74 | 1.75 | 1.00 |
| glimepiride_body | 0.329 | 0.326 | 0.99 | 1.24 | 1.27 | 1.01 |
| Smith_BMCSystBiol2013 | 0.636 | 0.624 | 0.98 | 18.9 | 19.0 | 1.00 |
| Chen_MSB2009 | 2.200 | 2.228 | 1.01 | 1545 | 1504 | 0.97 |

## What the prototype does not cover, and open problems for upstream

- **Fallbacks to the difference quotient:**
  - conserved moiety analysis;
  - stoichiometry set by rules;
  - species conversion factors;
  - a global conversion factor that depends on the state;
  - calls to functions without a derivative: `rateOf` (it reads the stoichiometry through `csr_matrix_get_nz`, which could be marked inactive since a fallback stoichiometry is constant), `factorial`, the `distrib` functions, `csc`, `sec`, `cot` and the other reciprocal trigonometric helpers;
  - models restored with `saveState`/`loadState`.

  Most of these can be supported: conserved moieties by differentiating with respect to the independent species, the helpers with IR bodies like `rr_min`/`rr_max`.
- **Load time**, see above.
- **Dependencies.** Enzyme has to match roadrunner's LLVM exactly, which ties every LLVM update to an Enzyme release. KLU needs SuiteSparse and a SUNDIALS built with it in libroadrunner-deps. Both have to be added to the dependency builds and the wheels on all platforms; the prototype only ran on Linux.
- **Not used yet:** `getFullJacobian`, `getReducedJacobian`, the steady-state solvers (NLEQ, KINSOL) and forward sensitivities could use the same Jacobian.
- **Not measured:** a sparse difference quotient with the same coloring would also feed KLU without Enzyme. It costs one model evaluation per color, 2 to 7 here, and a finite difference error. It would be the cheaper path to the KLU speedup if exactness and the Newton accuracy turn out not to matter.
- **API.** A combination of `jacobian` and `linear_solver` the model cannot serve falls back to the dense difference quotient with a warning, so the two settings can be set in any order. The one error is `jacobian = "enzyme"` on a model loaded without `Config.LLVM_ENZYME_JACOBIAN`. The two settings exist in every build, so they appear in `getSettings()` and in saved states, which `develop` does not know.
- **Python binding issues found along the way, outside the prototype:**
  - A model symbol named `reset` becomes a property of the `RoadRunner` class, which hides `RoadRunner.reset` (test suite cases 00952 to 00999).
  - A 1 x 1 matrix comes back from the bindings with fewer dimensions.
  - `python_tests_RunStochasticTestSuite` finds the test suite only at fixed relative paths.
  - About 110 `ModelEditingTests`/`StateSavingTests` fail under `ctest -j20`, also on `develop`.
