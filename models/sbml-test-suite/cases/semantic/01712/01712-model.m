(*

category:        Test
synopsis:        An event triggered at t0 that uses a parameter with the same ID as a local parameter.
componentTags:   Compartment, EventWithDelay, Parameter, Reaction, Species
testTags:        Amount, EventT0Firing, LocalParameters, NonConstantParameter
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, an event would be triggered at t0 if it was testing the local parameter instead of the global parameter.

The model contains:
* 1 species (S1)
* 1 parameter (k1)
* 1 local parameter (J0.k1)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S1 | $k1$ |]


There is one event:

[{width:40em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *initialValue*  |  *Delay*  | *Event Assignments* |
| E1 | $k1 >= 2.25$ | false | $2$ | $k1 = 2$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $0$ | variable |
| Initial value of parameter k1 | $0$ | variable |
| Initial value of local parameter 'J0.k1' | $3$ | constant |
| Initial volume of compartment 'C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
