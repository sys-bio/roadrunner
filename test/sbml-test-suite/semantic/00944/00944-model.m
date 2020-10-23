(*

category:      Test
synopsis:      Basic one reaction with one species in one compartment and one event, assigning to a parameter.
componentTags: Compartment, EventNoDelay, Parameter, Reaction, Species
testTags:      Amount, NonConstantParameter
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

This model tests when an event changes a parameter involved in a kinetic law.

The model contains:
* 1 species (S1)
* 1 parameter (k1)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| -> S1 | $C * k1 / S1$ |]


There is one event:

[{width:30em,margin: 1em auto}|  *Event*  |  *Trigger*  | *Event Assignments* |
| event1 | $S1 > 2.1$ | $k1 = 10$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $1$ | variable |
| Initial value of parameter k1 | $1$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

*)

newcase[ "00944" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.0];
addParameter[ k1, value -> 1 ];
addReaction[ -> S1, reversible -> False,
	     kineticLaw -> k1 * C / S1 ];
addEvent[ trigger -> S1 < 0.75, eventAssignment -> k1->10 ];

makemodel[]


*/

