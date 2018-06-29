(*

category:        Test
synopsis:        A simple reaction whose identifier is used in an event assignment.
componentTags:   CSymbolTime, Compartment, EventNoDelay, Parameter, Reaction, Species
testTags:        Amount, NonConstantParameter
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

This test was written to check using the identifier of a reaction (J0) in an event assignment.

The model contains:
* 1 species (S1)
* 1 parameter (k1)
* 1 compartment (c)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S1 | $k1$ |]


There is one event:

[{width:30em,margin: 1em auto}|  *Event*  |  *Trigger*  | *Event Assignments* |
| _E0 | $time > 5.5$ | $k1 = 1 + J0$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $0$ | variable |
| Initial value of parameter k1 | $1$ | variable |
| Initial volume of compartment 'c' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

