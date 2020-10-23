(* 

category:      Test
synopsis:      Basic two reactions with three species in one compartment with
with an algebraic rule used to determine value of the compartment size.
componentTags: Compartment, Species, Reaction, Parameter, AlgebraicRule 
testTags:      Amount, InitialValueReassigned
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are three
species called S1, S2 and S3 and two parameters called k1 and k2.  The model
contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 | $k1 * S1 * S2 * C$  |
| S3 -> S1 + S2 | $k2 * S3 * C$     |]

The model contains one rule which must be used to determine the value of
compartment C:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Algebraic |   n/a    | $C - 1$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*                   |
|Initial amount of S1                |$1.0 \x 10^-3$  |mole                      |
|Initial amount of S2                |$2.0 \x 10^-3$  |mole                      |
|Initial amount of S3                |$1.0 \x 10^-3$  |mole                      |
|Value of parameter k1               |$0.75$           |litre mole^-1^ second^-1^ |
|Value of parameter k2               |$0.25$           |second^-1^                |
|Volume of compartment C             |$undeclared$              |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00540" ];

addCompartment[ C, constant -> False ];
addSpecies[ S1, initialAmount -> 1.0 10^-3];
addSpecies[ S2, initialAmount -> 2.0 10^-3];
addSpecies[ S3, initialAmount -> 1.0 10^-3];
addParameter[ k1, value -> 0.75 ];
addParameter[ k2, value -> 0.25 ];
addRule[ type->AlgebraicRule, math -> C - 1];
addReaction[ S1 + S2 -> S3, reversible -> False,
	     kineticLaw -> k1 * S1 * S2 * C ];
addReaction[ S3 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 * C ];

makemodel[]

