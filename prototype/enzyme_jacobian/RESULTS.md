# Enzyme Jacobian and KLU for CVODE: results of the prototype

The question was whether it is worth bringing two things into roadrunner: the exact Jacobian, which Enzyme computes from roadrunner's generated LLVM IR, and the sparse KLU solver. The design is option 3 of sys-bio/roadrunner #1412, combined with #1409. The work is on branch `enzyme-jacobian` of the fork; `BUILD.md` has the build instructions. Every measurement was made with both of roadrunner's JITs, MCJit and LLJit.

## Answer

Yes. The prototype is worth pursuing:

- **Exact.** The Enzyme Jacobian matches finite differences:
  - on all four benchmark models, at three states each;
  - on all 1417 cases of the SBML test suite it covers.

  This holds with MCJit and with LLJit at the codegen levels `none` and `aggressive`. Every nonzero of the finite differences lies inside the sparsity pattern.
- **Fast per Jacobian.** One Enzyme Jacobian costs 2 to 50 times less than CVODE's dense difference quotient. The advantage grows with the size of the model.
- **Fast per simulation, but only with KLU.** With the dense solver, the Enzyme Jacobian makes a simulation at most 16 % faster, because CVODE evaluates a Jacobian only 6 to 27 times per simulation. With KLU, simulations are 1.2 to 5.3 times faster, and the gain grows with the model (Chen_MSB2009, 500 states: 4.7 to 5.3x). The choice of JIT does not change these ratios.
- **Loading the largest model costs a few seconds, not tens of seconds.** Most of the load time went into the Enzyme pass itself, not into the JIT. Differentiating the reaction rates in blocks of 64 reactions cuts the pass to a third. On the same loaded machine, Chen_MSB2009 now loads with the Enzyme Jacobian in:
  - 5.2 s with LLJit (1.1 s without the Enzyme Jacobian);
  - 8.4 s with MCJit (3.5 s without).

  The earlier MCJit measurement, before the blocks, took 38 s. For the three smaller models, the Enzyme Jacobian adds 0.2 to 0.7 s to loading.
- **No cost when switched off.** A build with Enzyme and KLU, with the feature switched off, loads and simulates within 3 % of `develop`.

## JIT backends

roadrunner has two JITs. Three of their properties shaped how I measured:

- **MCJit is still the default.** #1215 was meant to make LLJit the default. It changed the map of the configuration file names instead of the default value, so `Config.LLVM_BACKEND` is still MCJit, both in `develop` and in the 2.11 wheels (#1419). The LLJit defaults that are actually in effect are codegen level `none` and one compile thread; #1215 apparently intended `aggressive`.
- **LLJit runs no IR optimization passes.** Only MCJit runs the `OPTIMIZE_*` passes. With LLJit, the codegen level is the only optimization of the generated code, and that includes Enzyme's derivative code.
- **LLJit compiles a model only once per process.** It keeps the compiled object regardless of `LOADSBMLOPTIONS_RECOMPILE` and of the optimization level (#1420). This has two consequences:
  - A second load of the same model in a process includes no compilation, so timing it measures nothing about compilation.
  - Loading again with a different optimization level silently reuses the code of the first level.

  `benchmark.py` therefore loads every model in a fresh process and runs every backend in a process of its own.

I measured three configurations:

- `mcjit`;
- `lljit-none`, LLJit as it is configured today;
- `lljit-aggressive`, LLJit as #1215 meant it.

## Go/no-go gate

The gate passed with the tagged Enzyme release `v0.0.303`, built against roadrunner's LLVM 23.1.4, with no patches. Two things were needed:

- **IR bodies for some helper functions.** The generated code calls C++ helpers that Enzyme cannot see into: `rr_min`, `rr_max`, `rr_logd`, `rr_rootd`, `arccot` and `rr_arccot_negzero`. They get IR bodies in the module.
- **A diagnostic handler.** Enzyme reports failures as LLVM error diagnostics, which terminate the process by default. A diagnostic handler turns them into exceptions. A model that Enzyme cannot handle then falls back to the difference quotient instead of crashing.

## Validation (`validation.md`, `validation-lljit-none.md`, `validation-lljit-aggressive.md`)

- **Reference.** The Enzyme Jacobian is compared with finite differences of the state vector rate:
  - central differences over a ladder of step sizes, within 1e-6 relative;
  - one-sided differences within 1e-4 relative, for the kinks of `max` and `min`.
- **Entries the finite differences cannot judge.** Sometimes the central differences of the different steps disagree with each other by more than the closest of them differs from the Enzyme value. Then they are rounding noise and cannot judge the entry, so the entry passes and is counted separately.
  - This happens in one case only: 01208, with LLJit at `aggressive`.
  - The entry is zero by cancellation (`dA/dt = V dS/dt + S dV/dt` in a growing compartment).
  - The Enzyme Jacobian gives 2e-16 there, the finite differences up to 9e-11. Fused multiply-adds keep the generated rate from cancelling exactly.
- **Reported error.** The relative error is reported only over entries whose central differences agree with each other within 1e-3. The largest is 1.4e-6 (case 01395).
- **Benchmark models.** There are no mismatches and no finite difference nonzeros outside the sparsity pattern, with every backend. The largest relative error is 1.0e-9.
- **SBML test suite.** The results are the same with every backend. Of the 1823 semantic cases:
  - 1417 pass.
  - 164 fall back to the difference quotient, each with a reason:
    - 104 have a stoichiometry set by a rule.
    - 46 have species conversion factors.
    - 14 call a function that has no derivative: 10 call `csr_matrix_get_nz` (which `rateOf` uses to read a stoichiometry), 3 call `factorial` and 1 calls `sec`.
  - 242 are skipped. roadrunner cannot simulate 208 of them (algebraic rules, delays, fast reactions), and 34 have no timecourse.
  - None fail.
- **Blocks.** The test suite models have few reactions, so with the default of 64 reactions per block only the benchmark models are split: Chen_MSB2009 into 13 blocks, Smith_BMCSystBiol2013 into 6 and glimepiride_body into 2.
  - With one reaction per block (`RR_ENZYME_BLOCK_SIZE=1`, LLJit), 528 test suite cases are split into 2 to 52 blocks. All 1417 cases and the benchmark models still pass.
  - A C++ test checks that the Jacobian does not depend on the block size.
- **Simulations.** All three configurations stay within the integration error of a tight-tolerance reference (`rtol = 1e-12`, `atol = 1e-14`). In test case 00197, the difference quotient is the less accurate one.
- **Tests.** Every Enzyme and CVODE test runs with both JITs, 50 C++ tests in all.
- **Bugs found and fixed:**
  - The model cache key ignored the Enzyme option.
  - The sparsity pattern missed the amount conversion of rate rules on concentration species (16 test suite cases).
  - The final code review found a crash. Some models cannot serve the `jacobian`/`linear_solver` settings: a model that fell back, a model loaded without the Enzyme option, or a model with conserved moiety analysis. Loading or regenerating such a model made CVODE throw halfway through its setup, and the next simulation crashed. CVODE now falls back to the dense difference quotient, with a warning, on every such path, including copies of a RoadRunner.

## Benchmark (`jit.csv`)

All configurations come from the same release build. The simulation settings are `rtol = atol = 1e-10` and `simulate(0, end, 201)`.

- **How the times are taken.** Each backend runs in a process of its own, and the backends take turns for three rounds. Within a round, each configuration is simulated 10 times, with the configurations taking turns, and each model is loaded 3 times, each in a fresh process. Every time in the tables is the median over the rounds of the per-round medians.
- **Jacobian timings** are measured with `enzyme_jacobian_timing`, at the state in the middle of the timecourse. The difference quotient is counted as n evaluations of the state vector rate, which is what CVODE's dense difference quotient costs.
- **Shared machine.** Other jobs were running at the same time: load average 12 to 22 on an i7-12700 with 8 performance cores and 4 efficiency cores. A process that lands on an efficiency core runs 2 to 3 times slower. That is why single times in the table scatter, for example the LLJit `none` Jacobian of hctz_body. The ratios within a backend are stable within about 10 %.

`glimepiride_body` is `glimepiride_body_flat.xml` of `~/git/glimepiride-model/models` with 51 states. It is not the 168-state model of the #1409 profiles. `models.py` names every file and the PEtab commit.

| model | states | J nnz | colors |
| --- | ---: | ---: | ---: |
| hctz_body | 28 | 91 | 5 |
| glimepiride_body | 51 | 125 | 2 |
| Smith_BMCSystBiol2013 | 104 | 802 | 3 |
| Chen_MSB2009 | 500 | 4400 | 7 |

| model | backend | load (s) | load with Enzyme (s) | Jacobian, DQ (µs) | Jacobian, Enzyme (µs) | speedup | simulate, dense + DQ (ms) | dense + Enzyme | KLU + Enzyme |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hctz_body | mcjit | 0.28 | 0.56 | 11 | 5.86 | 1.9x | 2.21 | 2.15 (1.03x) | 1.9 (1.16x) |
| hctz_body | lljit-none | 0.12 | 0.30 | 25.4 | 12.3 | 2.1x | 2.08 | 1.95 (1.07x) | 1.73 (1.20x) |
| hctz_body | lljit-aggressive | 0.19 | 0.39 | 11.7 | 5.34 | 2.2x | 2.08 | 1.94 (1.07x) | 1.7 (1.23x) |
| glimepiride_body | mcjit | 0.44 | 0.85 | 12 | 1.42 | 8.5x | 2.49 | 2.61 (0.95x) | 1.91 (1.31x) |
| glimepiride_body | lljit-none | 0.15 | 0.54 | 18.5 | 2.34 | 7.9x | 3.2 | 3.01 (1.06x) | 2.49 (1.28x) |
| glimepiride_body | lljit-aggressive | 0.29 | 0.70 | 18.6 | 1.96 | 9.5x | 1.58 | 1.48 (1.06x) | 1.15 (1.38x) |
| Smith_BMCSystBiol2013 | mcjit | 0.94 | 1.68 | 72.2 | 4.4 | 16.4x | 19.7 | 17 (1.16x) | 11.4 (1.73x) |
| Smith_BMCSystBiol2013 | lljit-none | 0.32 | 0.87 | 74.8 | 4.82 | 15.5x | 22 | 18.9 (1.16x) | 12.6 (1.74x) |
| Smith_BMCSystBiol2013 | lljit-aggressive | 0.64 | 1.30 | 69.5 | 6.13 | 11.3x | 19.1 | 17.2 (1.11x) | 11.2 (1.70x) |
| Chen_MSB2009 | mcjit | 3.47 | 8.40 | 2570 | 53.1 | 48.4x | 1930 | 1837 (1.05x) | 414 (4.66x) |
| Chen_MSB2009 | lljit-none | 1.08 | 5.23 | 2826 | 64.2 | 44.0x | 1820 | 1855 (0.98x) | 379 (4.80x) |
| Chen_MSB2009 | lljit-aggressive | 1.70 | 7.17 | 2643 | 51.8 | 51.0x | 2144 | 2007 (1.07x) | 406 (5.29x) |

The next table gives the CVODE counters of one simulation. MCJit and LLJit at `none` take exactly the same steps. At `aggressive` the steps differ slightly, because fused multiply-adds round differently.

| model | backend | configuration | steps | RHS evaluations | RHS evaluations for the Jacobian | Jacobian evaluations | Newton iterations | Newton failures | error test failures | linear solver setups |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hctz_body | all | dense + DQ | 878 | 923 | 420 | 15 | 922 | 0 | 2 | 77 |
| hctz_body | all | dense + Enzyme, KLU + Enzyme | 878 | 923 | 0 | 15 | 922 | 0 | 2 | 77 |
| glimepiride_body | all | dense + DQ | 338 | 359 | 306 | 6 | 356 | 0 | 0 | 35 |
| glimepiride_body | all | dense + Enzyme, KLU + Enzyme | 338 | 359 | 0 | 6 | 356 | 0 | 0 | 35 |
| Smith_BMCSystBiol2013 | all | dense + DQ | 1177 | 1265 | 2080 | 20 | 1262 | 0 | 9 | 89 |
| Smith_BMCSystBiol2013 | mcjit, lljit-none | dense + Enzyme | 1199 | 1295 | 0 | 20 | 1292 | 0 | 6 | 85 |
| Smith_BMCSystBiol2013 | lljit-aggressive | dense + Enzyme | 1242 | 1369 | 0 | 21 | 1366 | 0 | 13 | 99 |
| Smith_BMCSystBiol2013 | all | KLU + Enzyme | 1177 | 1265 | 0 | 20 | 1262 | 0 | 9 | 89 |
| Chen_MSB2009 | mcjit, lljit-none | dense + DQ | 1349 | 2108 | 13000 | 26 | 2105 | 4 | 143 | 283 |
| Chen_MSB2009 | mcjit, lljit-none | dense + Enzyme | 1429 | 2188 | 0 | 27 | 2185 | 8 | 157 | 310 |
| Chen_MSB2009 | mcjit, lljit-none | KLU + Enzyme | 587 | 620 | 0 | 10 | 617 | 0 | 4 | 52 |
| Chen_MSB2009 | lljit-aggressive | dense + DQ | 1323 | 2079 | 12500 | 25 | 2076 | 3 | 138 | 274 |
| Chen_MSB2009 | lljit-aggressive | dense + Enzyme | 1330 | 2030 | 0 | 25 | 2027 | 5 | 131 | 271 |
| Chen_MSB2009 | lljit-aggressive | KLU + Enzyme | 608 | 657 | 0 | 11 | 654 | 0 | 7 | 57 |

## Where the speedup comes from, and where it does not

- **Fewer model evaluations, which matters little.** The Enzyme Jacobian removes every right-hand-side evaluation of the difference quotient, 31 to 86 % of all evaluations. But CVODE reuses its Jacobian and evaluates it only 6 to 27 times per simulation. On these models, evaluating the model is cheap next to the linear algebra, so the dense solver gains at most 16 %.
- **Sparse linear algebra, which matters most.** KLU factorises and solves the sparse `I - γJ` instead of a dense matrix. The gain grows with the model, as the profiles in #1409 predicted: from 1.2x at 28 states to about 5x at 500 states. KLU needs a Jacobian from roadrunner, so the Enzyme Jacobian is what makes KLU possible, rather than the source of the speedup itself.
- **More accurate Newton steps, which shows only on the largest model.**
  - The exact Jacobian alone does not reduce the step counts. Dense + Enzyme takes the same steps as dense + DQ on the PBPK models, and slightly more on Smith and Chen.
  - On Chen_MSB2009, KLU + Enzyme takes 587 steps instead of 1349, with 4 error test failures instead of 143. The dense solver with the exact Jacobian behaves like the dense difference quotient, so the gain comes from the linear solver.
  - My hypothesis is that dense LU loses accuracy on this badly scaled matrix (values up to 1.2e9), and KLU, which scales the rows, does not. I have not verified it. It is worth a closer look, because it decides whether users of the dense solver would gain from a better-conditioned solve too.
- **The JIT matters for the derivative code, not for the ratios.**
  - LLJit at `none` runs no IR optimization, so it turns Enzyme's derivative code into slower machine code. On Chen_MSB2009 a Jacobian takes 64 µs, against 52 to 53 µs with MCJit or with LLJit at `aggressive`.
  - That is still 44 times faster than the difference quotient. The simulations hardly notice the difference, since the Jacobian is evaluated rarely.
- **The vector mode does not pay off.** The time of a Jacobian hardly depends on the vector width, while the compilation time grows with it. The width is therefore 1 by default, one seed direction per pass; the environment variable `RR_ENZYME_MAX_WIDTH` sets a larger one. These numbers were measured with MCJit, before the blocks, with other jobs on the machine (`width.csv`):

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

## Load time (`blocks.csv`)

**Where the time went.** With MCJit, the first version of the prototype loaded Chen_MSB2009 in 38 s instead of 6 s on the loaded machine, and compiling the derivative code looked like the cause. LLJit at `none` compiles the same code in about 1 s. That showed the Enzyme pass itself was the larger part: 8.6 s for Chen_MSB2009, whose reaction rates are one function of 827 reactions.

- Sampling the pass with gdb put most of the time into Enzyme's activity analysis and type analysis.
- Both grow faster than linearly with the size of the function. The pass takes 0.14 s for hctz_body (31 reactions), 1.1 s for Smith_BMCSystBiol2013 (367 reactions) and 8.6 s for Chen_MSB2009 (827 reactions).
- Enzyme's command line options (`enzyme-preopt`, `enzyme-strict-aliasing`, `enzyme-global-activity`) did not help.

**The fix: blocks.** The prototype therefore differentiates `evalReactionRates` in blocks:

- Each block is a copy of the function in which only the stores of the block's reaction rates remain, with the code they no longer need removed.
- Enzyme differentiates each copy, and the tangents function calls the derivatives of all blocks.
- This requires that the function writes nothing but the reaction rates and never reads them back. That holds because a reaction id in the math is computed again from its kinetic law. The condition is checked on the IR; if it does not hold, the function is differentiated whole.

The table shows Chen_MSB2009 at different block sizes. Each value is the median of 3 loads in fresh processes, with other jobs running at the same time.

| backend | reactions per block | blocks | IR setup (s) | Enzyme pass (s) | compilation (s) | load with Enzyme (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mcjit | all | 1 | 0.01 | 11.5 | 10.7 | 23.2 |
| mcjit | 256 | 4 | 0.13 | 4.1 | 6.0 | 10.8 |
| mcjit | 128 | 7 | 0.29 | 3.3 | 5.4 | 9.7 |
| mcjit | 64 | 13 | 0.54 | 3.2 | 4.9 | 9.4 |
| mcjit | 32 | 26 | 1.10 | 3.5 | 6.1 | 13.4 |
| mcjit | 16 | 52 | 1.86 | 3.6 | 4.1 | 10.5 |
| lljit-none | all | 1 | 0.01 | 8.6 | 0.9 | 10.1 |
| lljit-none | 256 | 4 | 0.14 | 4.0 | 1.0 | 5.8 |
| lljit-none | 128 | 7 | 0.34 | 4.4 | 1.1 | 6.8 |
| lljit-none | 64 | 13 | 0.43 | 2.6 | 1.0 | 4.7 |
| lljit-none | 32 | 26 | 1.12 | 3.0 | 1.3 | 6.2 |
| lljit-none | 16 | 52 | 2.17 | 3.9 | 0.9 | 8.0 |

**The default block size is 64 reactions.**

- Smaller blocks cost more than they save: each block starts from a copy of the whole function, and each Enzyme call has its own overhead.
- The time of a Jacobian does not depend on the block size: 53 to 66 µs on performance cores.
- The copies are deleted after the Enzyme pass, so they are never compiled.

Options that remain for an upstream version:

- Copy only the instructions a block needs, instead of copying the whole function and deleting the rest. That removes most of the IR setup.
- Compile the derivative functions lazily, on the first Jacobian CVODE asks for, or in the background (LLJit can do this). A model that is never simulated with the Enzyme Jacobian then does not pay for it.
- Generate the reaction rates in blocks in the code generator itself. That would also allow blocks for `evalRateRuleRates`.

## Cost when switched off (`baseline.csv`)

This compares `develop` (`2876ab2ed`) with the feature build with the feature switched off. Both were built against the same dependencies and run alternately twice, with MCJit; with the feature off, the code path is the same for both JITs. The values are medians:

| model | load, develop (s) | load, feature (s) | ratio | simulate, develop (ms) | simulate, feature (ms) | ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hctz_body | 0.208 | 0.209 | 1.00 | 1.74 | 1.75 | 1.00 |
| glimepiride_body | 0.329 | 0.326 | 0.99 | 1.24 | 1.27 | 1.01 |
| Smith_BMCSystBiol2013 | 0.636 | 0.624 | 0.98 | 18.9 | 19.0 | 1.00 |
| Chen_MSB2009 | 2.200 | 2.228 | 1.01 | 1545 | 1504 | 0.97 |

## What the prototype does not cover, and open problems for upstream

- **Fallbacks to the difference quotient.** These cases still use the difference quotient:
  - conserved moiety analysis;
  - stoichiometry set by rules;
  - species conversion factors;
  - a global conversion factor that depends on the state;
  - calls to functions without a derivative:
    - `rateOf`, which reads the stoichiometry through `csr_matrix_get_nz`. That call could be marked inactive, since a stoichiometry is constant in every model that does not fall back for it.
    - `factorial`;
    - the `distrib` functions;
    - `csc`, `sec`, `cot` and the other reciprocal trigonometric helpers.
  - models restored with `saveState`/`loadState`.

  Most of these can be supported. Conserved moieties can be handled by differentiating with respect to the independent species, and the helpers can get IR bodies like `rr_min`/`rr_max`.
- **Load time.** See above: a few seconds for a model of 800 reactions.
- **Dependencies.**
  - Enzyme has to match roadrunner's LLVM exactly, which ties every LLVM update to an Enzyme release.
  - KLU needs SuiteSparse, and a SUNDIALS built with it, in libroadrunner-deps.
  - Both have to be added to the dependency builds and the wheels on all platforms. The prototype only ran on Linux.
- **Not used yet.** `getFullJacobian`, `getReducedJacobian`, the steady-state solvers (NLEQ, KINSOL) and forward sensitivities could all use the same Jacobian.
- **Not measured.** A sparse difference quotient with the same coloring would also feed KLU, without Enzyme. It costs one model evaluation per color (2 to 7 here) and has a finite difference error. It would be the cheaper path to the KLU speedup if exactness and the Newton accuracy turn out not to matter.
- **API.**
  - If the model cannot serve a combination of `jacobian` and `linear_solver`, CVODE falls back to the dense difference quotient with a warning, so the two settings can be set in any order.
  - The one error is `jacobian = "enzyme"` on a model loaded without `Config.LLVM_ENZYME_JACOBIAN`.
  - The two settings exist in every build, so they appear in `getSettings()` and in saved states, which `develop` does not know about.
- **Issues found along the way, outside the prototype:**
  - #1415: a model id named like a method of `RoadRunner`, e.g. `reset` (18 test suite cases), replaces that method for every instance and removes it for the rest of the process.
  - #1416: `python_tests_RunStochasticTestSuite` finds the test suite only at fixed relative paths.
  - #1417: about 110 `ModelEditingTests`/`StateSavingTests` fail under `ctest -j20`, also on `develop`, because they share one output directory.
  - #1419: MCJit is still the default JIT, and three keys of the configuration file map to the wrong settings.
  - #1420: LLJit reuses the compiled object of a model within a process, despite `LOADSBMLOPTIONS_RECOMPILE` and a changed optimization level.
  - Not a bug, but worth knowing for the new API: the bindings return a matrix with one column and no column names as a 1D array. The unnamed `getStateVectorJacobian()` of a model with one state therefore comes back with shape `(1,)`, as documented in `matrix_to_array`.
