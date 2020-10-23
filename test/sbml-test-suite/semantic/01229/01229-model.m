(*

category:        Test
synopsis:        A simple reaction whose identifier is used in an event priority.
componentTags:   CSymbolTime, Compartment, EventNoDelay, EventPriority, Parameter, Reaction, Species
testTags:        Amount, NonConstantParameter
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

This test was written to check using the identifier of a reaction (J0) in an event priority.  E1 has a lower priority, and thus executes after E0, leaving k1 with a value of 2 instead of 10.

The model contains:
* 1 species (S1)
* 1 parameter (k1)
* 1 compartment (c)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S1 | $k1$ |]


There are 2 events:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  | *Event Assignments* |
| E0 | $time > 5.5$ | $J0$ | $k1 = 10$ |
| E1 | $time > 5.5$ | $J0 - 1$ | $k1 = 2$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $0$ | variable |
| Initial value of parameter k1 | $1$ | variable |
| Initial volume of compartment 'c' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

