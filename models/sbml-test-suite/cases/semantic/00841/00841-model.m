(* 

category:      Test
synopsis:      Single reversible reaction.
componentTags: Compartment, Species, Reaction, Parameter, RateRule 
testTags:      Amount
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are four species called 
S1, S2, S3 and S4 and two parameters called kf and kr.  The model contains  one reaction
defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 <-> S2 + S3 | $(kf * S1 - kr * S2 * S3) * C$  |]

The model contains one rule which specifies the rate at which species 
S4 is varying:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Rate | S4 | $-0.5 * (kf * S1 - kr * S2 * S3)$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value*       |*Units*  |
|Initial amount of S1        |$1.0$ |mole                      |
|Initial amount of S2        |$0.5$ |mole                      |
|Initial amount of S3        |$0$ |mole                      |
|Initial amount of S4        |$0.5$ |mole                      |
|Value of parameter kf       |$2.5$          |second^-1^ |
|Value of parameter kr       |$0.2$          |litre mole^-1^ second^-1^ |
|Volume of compartment C     |$1$             |litre                  |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00841" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.0];
addSpecies[ S2, initialAmount -> 0.5];
addSpecies[ S3, initialAmount -> 0.0];
addSpecies[ S4, initialAmount -> 0.5];
addParameter[ kf, value -> 2.5 ];
addParameter[ kr, value -> 0.2 ];
addRule[ type->RateRule, variable -> S4, math -> -0.5 * (kf * S1 - kr * S2 * S3)];
addReaction[ S1 -> S2 + S3,
	     kineticLaw -> (kf * S1 - kr * S2 * S3) * C ];

makemodel[]

