(*

category:        Test
synopsis:        A delayed event using assignment-time values.
componentTags:   Compartment, EventWithDelay, Parameter, Reaction, Species
testTags:        Amount, EventUsesAssignmentTimeValues, NonConstantParameter
testType:        TimeCourse
levels:          2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, an event uses assignment-time values, which makes a difference to the results.  This model is the same as 01325, but with a reaction instead of a rate rule.

The model contains:
* 1 species (x)
* 1 parameter (y)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> x | $1$ |]


There is one event:

[{width:40em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Use values from:*  |  *Delay*  | *Event Assignments* |
| E0 | $x >= 3.5$ | Assignment time | $2$ | $y = x$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species x | $1$ | variable |
| Initial value of parameter y | $0$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
