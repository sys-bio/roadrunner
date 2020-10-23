(*

category:      Test
synopsis:      Basic one reaction with one species in one compartment and one event, assigning to a parameter.
componentTags: AssignmentRule, Compartment, EventNoDelay, Parameter, Reaction, Species
testTags:      Concentration, NonConstantCompartment, NonConstantParameter, NonUnityCompartment, InitialValueReassigned
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There is one species called S1 and two parameters, k1 and fakeC.  It is identical in design and expected output to model 00945, except that instead of the event assigning to a compartment directly, it assigns to a temporary variable 'fakeC' which is then assigned to compartment C through an assignment rule.

The model contains:
* 1 species (S1)
* 2 parameters (fakeC, k1)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| -> S1 | $C * k1 / S1$ |]


The model contains one event that assigns a value to the temporary parameter fakeC:

[{width:30em,margin: 1em auto}|  *Event*  |  *Trigger*  | *Event Assignments* |
| event1 | $S1 > 2.1$ | $fakeC = 10$ |]


And the model contains one assignment rule that assigns the value of fakeC to compartment C:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | C | $fakeC$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $1$ | variable |
| Initial value of parameter k1 | $1$ | constant |
| Initial value of parameter fakeC | $1$ | variable |
| Initial volume of compartment 'C' | $fakeC$ | variable |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

*)

newcase[ "00946" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.0];
addParameter[ k1, value -> 1 ];
addParameter[ fakeC, value -> 1 ];
addRule[ type -> assignmentRule, variable -> k1, math -> multiply[p1, p2]];
addReaction[ -> S1, reversible -> False,
	     kineticLaw -> k1 * C / S1 ];
addEvent[ trigger -> S1 < 0.75, eventAssignment -> C->10 ];

makemodel[]


*/

