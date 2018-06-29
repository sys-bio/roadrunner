(*

category:        Test
synopsis:        Single objective, test: R07 EQ 0.2, R10 EQ 0.2, R25 EQ 0.6
componentTags:   Compartment, Parameter, Reaction, Species, fbc:FluxObjective, fbc:Objective
testTags:        BoundaryCondition, NonUnityStoichiometry, fbc:MaximizeObjective
testType:        FluxBalanceSteadyState
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: fbc, fbc_v2

 Single objective, test: R07 EQ 0.2, R10 EQ 0.2, R25 EQ 0.6

The model contains:
* 23 species (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, X, Y)
* 7 parameters (fb_0, fb_1000, fb_n1000, fb_1, fb_0_2, fb_0_3, fb_0_4)
* 1 compartment (Cell)
* 1 objective (OBJF)

The active objective is OBJF, which is to be maximized:
  + 1 R26

There are 26 reactions:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| R16: J -> M | $fb_0 <= R16 <= fb_1000$$ |
| R14: K -> L | $fb_n1000 <= R14 <= fb_1000$$ |
| R15: L -> Q | $fb_0 <= R15 <= fb_1000$$ |
| R12: I -> J | $fb_0 <= R12 <= fb_1000$$ |
| R07: E -> F | $fb_0_2 <= R07 <= fb_0_2$$ |
| R13: J -> K | $fb_0 <= R13 <= fb_1000$$ |
| R01: X -> A | $fb_0 <= R01 <= fb_1$$ |
| R17: M -> N | $fb_0 <= R17 <= fb_1000$$ |
| R03: A -> C | $fb_n1000 <= R03 <= fb_1000$$ |
| R02: A -> B | $fb_n1000 <= R02 <= fb_1000$$ |
| R05: B -> D | $fb_0 <= R05 <= fb_1000$$ |
| R04: C -> B | $fb_n1000 <= R04 <= fb_1000$$ |
| R10: G -> H | $fb_0_3 <= R10 <= fb_0_3$$ |
| R06: D -> E | $fb_0 <= R06 <= fb_1000$$ |
| R09: D -> G | $fb_0 <= R09 <= fb_1000$$ |
| R08: F -> I | $fb_0 <= R08 <= fb_1000$$ |
| R11: H -> I | $fb_0 <= R11 <= fb_1000$$ |
| R18: N -> Q | $fb_0 <= R18 <= fb_1000$$ |
| R19: K -> O | $fb_n1000 <= R19 <= fb_1000$$ |
| R26: S -> Y | $fb_0 <= R26 <= fb_1000$$ |
| R25: A + T -> 0.5S + U | $fb_0_4 <= R25 <= fb_0_4$$ |
| R24: R -> S | $fb_0 <= R24 <= fb_1000$$ |
| R23: R -> S | $fb_n1000 <= R23 <= fb_1000$$ |
| R22: Q -> R | $fb_0 <= R22 <= fb_1000$$ |
| R21: P -> L | $fb_n1000 <= R21 <= fb_1000$$ |
| R20: O -> P | $fb_n1000 <= R20 <= fb_1000$$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter fb_0 | $0$ | constant |
| Initial value of parameter fb_1000 | $1000$ | constant |
| Initial value of parameter fb_n1000 | $-1000$ | constant |
| Initial value of parameter fb_1 | $1$ | constant |
| Initial value of parameter fb_0_2 | $0.2$ | constant |
| Initial value of parameter fb_0_3 | $0.3$ | constant |
| Initial value of parameter fb_0_4 | $0.4$ | constant |]

*)
