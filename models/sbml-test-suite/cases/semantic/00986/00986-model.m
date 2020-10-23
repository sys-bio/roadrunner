(*

category:      Test
synopsis:      A fast and slow reaction, the first of which feeds into the second.
componentTags: Compartment, Reaction, Species
testTags:      Amount, FastReaction, HasOnlySubstanceUnits
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1
generatedBy:   Analytic

The 'fast' reaction should go to completion instantly, converting all of A to B.  Then B converts to C normally.

The model contains:
* 3 species (A, B, C)
* 1 compartment (default_compartment)

There are 2 reactions:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |  *Fast*  |
| A -> B | $1 * A$ | fast |
| B -> C | $1 * B$ | slow |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species A | $1$ | variable |
| Initial amount of species B | $0$ | variable |
| Initial amount of species C | $0$ | variable |
| Initial volume of compartment 'default_compartment' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
