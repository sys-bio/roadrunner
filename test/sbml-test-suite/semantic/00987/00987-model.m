(*

category:      Test
synopsis:      A fast and slow reaction, the first of which feeds into the second, triggered by an event.
componentTags: CSymbolTime, Compartment, EventNoDelay, Parameter, Reaction, Species
testTags:      Amount, FastReaction, NonConstantParameter, HasOnlySubstanceUnits
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1
generatedBy:   Analytic

After the event fires, the 'fast' reaction should go to completion instantly, converting all of A to B.  Then B converts to C normally.

The model contains:
* 3 species (A, B, C)
* 1 parameter (k1)
* 1 compartment (default_compartment)

There are 2 reactions:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |  *Fast*  |
| A -> B | $k1 * A$ | fast |
| B -> C | $k1 * B$ | slow |]


There is one event:

[{width:30em,margin: 1em auto}|  *Event*  |  *Trigger*  | *Event Assignments* |
| _E0 | $time > 0.99$ | $k1 = 1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species A | $1$ | variable |
| Initial amount of species B | $0$ | variable |
| Initial amount of species C | $0$ | variable |
| Initial value of parameter k1 | $0$ | variable |
| Initial volume of compartment 'default_compartment' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
