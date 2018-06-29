(*

category:        Test
synopsis:        Two simultaneous events with a priority with the same ID as a local parameter.
componentTags:   CSymbolTime, Compartment, EventNoDelay, EventPriority, Parameter, Reaction, Species
testTags:        Amount, LocalParameters, NonConstantParameter
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, the two simultaneous events fire in one order because of the priorities, but would fire in the wrong order if a local parameter was confused for the global parameter.

The model contains:
* 1 species (S1)
* 1 parameter (k1)
* 1 local parameter (J0.k1)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S1 | $k1$ |]


There are 2 events:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  | *Event Assignments* |
| E0 | $time > 2.5$ | $k1$ | $k1 = 5$ |
| E1 | $time > 2.5$ | $2.5$ | $k1 = 10$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $0$ | variable |
| Initial value of parameter k1 | $2$ | variable |
| Initial value of local parameter 'J0.k1' | $3$ | constant |
| Initial volume of compartment 'C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
