(* 

category:      Test
synopsis:      Algebraic rule used to determine rate of species change.
componentTags: Compartment, Species, Reaction, Parameter, AlgebraicRule, EventWithDelay, InitialAssignment 
testTags:      Amount, InitialValueReassigned
testType:      TimeCourse
levels:        2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are four
species called X0, X1, T and S1 and two parameters called k1 and k2.  The model
contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| X0 -> T     | $C * k1 * X0$  |
| T -> X1     | $C * k2 * T$  |]

The model contains one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Algebraic |   n/a    | $X0 + X1 + T - S1$  |]

The model contains one event that assigns value to species X1 defined as:

[{width:30em,margin: 1em auto}| | *Trigger*    | *Delay* | *Assignments* |
 | Event1 | $X1 > 2$ | $4.3$   | $X1 = 1$    |]

The model contains one initialAssignment:

[{width:30em,margin: 1em auto}| Variable | Formula |
 | X0       | $X1 - 0.25$  |]

Note: SBML's InitialAssignment construct override any declared initial
values.  In this model, the initial value for the species X0 has not been
explicitly declared and must be calculated using the InitialAssignment.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value* |*Units*  |
|Initial amount of X0          |$undeclared$  |mole                      |
|Initial amount of X1          |$1.5$   |mole                      |
|Initial amount of T           |$1$     |mole                      |
|Initial amount of S1          |$3.75$  |mole                      |
|Value of parameter k1         |$0.1$   |second^-1^            |
|Value of parameter k2         |$0.2$   |second^-1^                |
|Volume of compartment C       |$1$     |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00780" ];

addCompartment[ C, size -> 1 ];
addSpecies[ X0];
addSpecies[ X1, initialAmount -> 1.5];
addSpecies[ T, initialAmount -> 1];
addSpecies[ S1, initialAmount -> 3.75];
addParameter[ k1, value -> 0.1 ];
addParameter[ k2, value -> 0.2 ];
addInitialAssignment[ X0, math -> X1 -0.25];
addRule[ type->AlgebraicRule, math -> X0 + X1 + T - S1];
addReaction[ X0 -> T, reversible -> False,
	     kineticLaw -> C * k1 * X0 ];
addReaction[ T -> X1, reversible -> False,
	     kineticLaw -> C * k2 * T ];
addEvent[ trigger -> X1 > 2, delay->4.3, eventAssignment -> X1->1];

makemodel[]

