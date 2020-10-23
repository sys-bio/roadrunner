(* 

category:      Test
synopsis:      Basic two reactions with five species in one compartment with
an algebraic rule used to determine rate of change of one species.
componentTags: Compartment, Species, Reaction, Parameter, AlgebraicRule 
testTags:      Amount, ConstantSpecies
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are five
species called S1, S2, S3, S4 and S5 and two parameters called k1 and k2.
Species S5 is labeled as a constant species.  The model
contains two reactions defined as:


[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 | $k1 * S1 * S2 * C$  |
| S3 -> S1 + S2 | $k2 * S3 * C$     |]

The model contains one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Algebraic |   n/a    | $S4 - S1 - S5$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*                   |
|Initial amount of S1                |$5.0 \x 10^-4$  |mole                      |
|Initial amount of S2                |$2.0 \x 10^-3$  |mole                      |
|Initial amount of S3                |$1.0 \x 10^-3$  |mole                      |
|Initial amount of S4                |$1.0 \x 10^-3$  |mole                      |
|Initial amount of S5                |$5.0 \x 10^-4$  |mole                      |
|Value of parameter k1               |$0.75$           |litre mole^-1^ second^-1^ |
|Value of parameter k2               |$0.25$           |second^-1^                |
|Volume of compartment C             |$1$              |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00556" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialAmount -> 0.5 10^-3];
addSpecies[ S2, initialAmount -> 2.0 10^-3];
addSpecies[ S3, initialAmount -> 1.0 10^-3];
addSpecies[ S4, initialAmount -> 1.0 10^-3];
addSpecies[ S5, initialAmount -> 0.5 10^-3, constant -> True];
addParameter[ k1, value -> 0.75 ];
addParameter[ k2, value -> 0.25 ];
addRule[ type->AlgebraicRule, math -> S4 - S1 - S5];
addReaction[ S1 + S2 -> S3, reversible -> False,
	     kineticLaw -> k1 * S1 * S2 * C ];
addReaction[ S3 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 * C ];

makemodel[]

