(*

category:        Test
synopsis:        FBC test with initial assignment of infinity.
componentTags:   Compartment, InitialAssignment, Parameter, Reaction, Species, fbc:FluxObjective, fbc:Objective
testTags:        BoundaryCondition, InitialValueReassigned, NonUnityStoichiometry, fbc:MaximizeObjective, fbc:NonStrict
testType:        FluxBalanceSteadyState
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: fbc, fbc_v2

 In this test, an initial assignment to a flux bound sets a value of infinity.

The model contains:
* 23 species (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, X, Y)
* 4 parameters (fb_0, fb_1, fb_inf, fb_ninf)
* 1 compartment (Cell)
* 1 objective (OBJF)

The active objective is OBJF, which is to be maximized:
  + 1 R26

There are 26 reactions:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| R16: J -> M | $fb_0 <= R16 <= fb_inf$$ |
| R14: K -> L | $fb_ninf <= R14 <= fb_inf$$ |
| R15: L -> Q | $fb_0 <= R15 <= fb_inf$$ |
| R12: I -> J | $fb_0 <= R12 <= fb_inf$$ |
| R07: E -> F | $fb_0 <= R07 <= fb_inf$$ |
| R13: J -> K | $fb_0 <= R13 <= fb_inf$$ |
| R01: X -> A | $fb_0 <= R01 <= fb_1$$ |
| R17: M -> N | $fb_0 <= R17 <= fb_inf$$ |
| R03: A -> C | $fb_ninf <= R03 <= fb_inf$$ |
| R02: A -> B | $fb_ninf <= R02 <= fb_inf$$ |
| R05: B -> D | $fb_0 <= R05 <= fb_inf$$ |
| R04: C -> B | $fb_ninf <= R04 <= fb_inf$$ |
| R10: G -> H | $fb_0 <= R10 <= fb_inf$$ |
| R06: D -> E | $fb_0 <= R06 <= fb_inf$$ |
| R09: D -> G | $fb_0 <= R09 <= fb_inf$$ |
| R08: F -> I | $fb_0 <= R08 <= fb_inf$$ |
| R11: H -> I | $fb_0 <= R11 <= fb_inf$$ |
| R18: N -> Q | $fb_0 <= R18 <= fb_inf$$ |
| R19: K -> O | $fb_ninf <= R19 <= fb_inf$$ |
| R26: S -> Y | $fb_0 <= R26 <= fb_inf$$ |
| R25: A + T -> 0.5S + U | $fb_0 <= R25 <= fb_inf$$ |
| R24: R -> S | $fb_0 <= R24 <= fb_inf$$ |
| R23: R -> S | $fb_ninf <= R23 <= fb_inf$$ |
| R22: Q -> R | $fb_0 <= R22 <= fb_inf$$ |
| R21: P -> L | $fb_ninf <= R21 <= fb_inf$$ |
| R20: O -> P | $fb_ninf <= R20 <= fb_inf$$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter fb_0 | $0$ | constant |
| Initial value of parameter fb_1 | $1$ | constant |
| Initial value of parameter fb_inf | $INF$ | constant |
| Initial value of parameter fb_ninf | $-inf$ | constant |]

*)
