(*

category:        Test
synopsis:        A reaction whose rate goes negative due to an event.
componentTags:   CSymbolTime, Compartment, EventNoDelay, Parameter, Reaction, Species
testTags:        Amount, NonConstantParameter, ReversibleReaction
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, a reaction that produces S1 starts off positive, but then goes negative, due to an event.

The model contains:
* 1 species (S1)
* 1 parameter (k1)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S1 | $k1$ |]


There is one event:

[{width:30em,margin: 1em auto}|  *Event*  |  *Trigger*  | *Event Assignments* |
| _E0 | $time > 5.5$ | $k1 = -1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $0$ | variable |
| Initial value of parameter k1 | $1$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
