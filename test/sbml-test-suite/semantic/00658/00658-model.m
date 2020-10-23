(* 

category:      Test
synopsis:      Algebraic rule used to determine rate of species change.
componentTags: Compartment, Species, Reaction, Parameter, AlgebraicRule, InitialAssignment 
testTags:      Amount, InitialValueReassigned
testType:      TimeCourse
levels:        2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are two
species called S1 and S2 and two parameters called k1 and k2.  The model
contains one reaction defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| ->S1     | $C * k2 * S2$  |]

The model contains one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Algebraic |   n/a    | $S1 + S2-k1$  |]

The model contains one initialAssignment that assigns the initial value for
parameter k1:

[{width:30em,margin: 1em auto}| Variable    | Formula      |
 | k1 | $1$  |]

Note: InitialAssignments override any declared initial values.  In this
case the value from the initialAssignment is consistent with the value
attributed to k1 by the model definition.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*         |*Units*  |
|Initial amount of S1                |$          0.5$ |mole                      |
|Initial amount of S2                |$          0.5$ |mole                      |
|Value of parameter k1               |$          1.0$ |mole litre^-1^            |
|Value of parameter k2               |$            1$ |second^-1^                |
|Volume of compartment C |$            1$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00658" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialAmount -> 0.5];
addSpecies[ S2, initialAmount -> 0.5];
addParameter[ k1, value -> 1.0];
addParameter[ k2, value -> 1 ];
addInitialAssignment[ k1, math -> 1];
addRule[ type->AlgebraicRule, math -> S1 + S2-k1];
addReaction[ products->{S1}, modifiers->S2, reversible -> False,
	     kineticLaw -> C * k2 * S2 ];

makemodel[]

