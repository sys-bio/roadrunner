(*

category:        Test
synopsis:        Single objective, test R07 LE 0.2, R10 LE 0.3
componentTags:   Compartment, Reaction, Species, fbc:FluxBound, fbc:FluxObjective, fbc:Objective
testTags:        BoundaryCondition, NonUnityStoichiometry, fbc:BoundGreaterEqual, fbc:BoundLessEqual, fbc:MaximizeObjective
testType:        FluxBalanceSteadyState
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: fbc, fbc_v1

 Single objective, test R07 LE 0.2, R10 LE 0.3

The model contains:
* 23 species (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, X, Y)
* 1 compartment (Cell)
* 1 objective (OBJF)

The active objective is OBJF, which is to be maximized:
  + 1 R26

There are 26 reactions, and 52 flux bounds:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| R01: X -> A | $R01 >= 0 && R01 <= 1$ |
| R02: A -> B | $R02 >= -1000 && R02 <= 1000$ |
| R03: A -> C | $R03 >= -1000 && R03 <= 1000$ |
| R04: C -> B | $R04 >= -1000 && R04 <= 1000$ |
| R05: B -> D | $R05 >= 0 && R05 <= 1000$ |
| R06: D -> E | $R06 >= 0 && R06 <= 1000$ |
| R07: E -> F | $R07 >= 0 && R07 <= 0.2$ |
| R08: F -> I | $R08 >= 0 && R08 <= 1000$ |
| R09: D -> G | $R09 >= 0 && R09 <= 1000$ |
| R10: G -> H | $R10 >= 0 && R10 <= 0.3$ |
| R11: H -> I | $R11 >= 0 && R11 <= 1000$ |
| R12: I -> J | $R12 >= 0 && R12 <= 1000$ |
| R13: J -> K | $R13 >= 0 && R13 <= 1000$ |
| R14: K -> L | $R14 >= -1000 && R14 <= 1000$ |
| R15: L -> Q | $R15 >= 0 && R15 <= 1000$ |
| R16: J -> M | $R16 >= 0 && R16 <= 1000$ |
| R17: M -> N | $R17 >= 0 && R17 <= 1000$ |
| R18: N -> Q | $R18 >= 0 && R18 <= 1000$ |
| R19: K -> O | $R19 >= -1000 && R19 <= 1000$ |
| R20: O -> P | $R20 >= -1000 && R20 <= 1000$ |
| R21: P -> L | $R21 >= -1000 && R21 <= 1000$ |
| R22: Q -> R | $R22 >= 0 && R22 <= 1000$ |
| R23: R -> S | $R23 >= -1000 && R23 <= 1000$ |
| R24: R -> S | $R24 >= 0 && R24 <= 1000$ |
| R25: A + T -> 0.5S + U | $R25 >= 0 && R25 <= 1000$ |
| R26: S -> Y | $R26 >= 0 && R26 <= 1000$ |]

*)

