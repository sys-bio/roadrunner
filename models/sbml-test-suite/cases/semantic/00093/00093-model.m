(* 

category:      Test
synopsis:      Two reactions with four species in one
compartment using an assignmentRule and a rateRule to vary species.
componentTags: Compartment, Species, Reaction, Parameter, RateRule, AssignmentRule 
testTags:      Amount, InitialValueReassigned
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called "compartment".  There are four
species named S1, S2, S3 and S and three parameters named k1, k2 and k3.
The model contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S3 | $k1 * S1 * compartment$  |
| S3 -> S1 | $k2 * S3 * compartment$  |]

The model contains two rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Assignment | S4 | $k3 * S1$  |
 | Rate       | S2 | $k2 * S3-k1 * S1$  |]

Note that in this case the initial value of species S4 is not explicitly
declared and must be calculated by the assignmentRule.  

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1                |$   1 \x 10^-5$ |mole                      |
|Initial amount of S2                |$ 1.5 \x 10^-5$ |mole                      |
|Initial amount of S3                |$   1 \x 10^-5$ |mole                      |
|Initial amount of S4                |$   undeclared$ |mole                      |
|Value of parameter k1               |$        0.015$ |second^-1^ |
|Value of parameter k2               |$          0.5$ |second^-1^ |
|Value of parameter k3               |$          1.5$ |dimensionless |
|Volume of compartment "compartment" |$            1$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00093" ];

addCompartment[ compartment, size -> 1];
addSpecies[ S1, initialAmount->1 10^-5 ];
addSpecies[ S2, initialAmount -> 1.5 10^-5];
addSpecies[ S3, initialAmount -> 1 10^-5];
addSpecies[ S4];
addParameter[ k1, value -> .015];
addParameter[ k2, value -> .5 ];
addParameter[ k3, value -> 1.5 ];
addRule[ type->AssignmentRule, variable -> S4, math ->k3 * S1];
addRule[ type->RateRule, variable -> S2, math ->k2 * S3-k1 * S1];
addReaction[ S1 -> S3, reversible -> False,
	     kineticLaw -> k1 * S1 * compartment ];
addReaction[ S3 -> S1, reversible -> False,
	     kineticLaw -> k2 * S3 * compartment ];

makemodel[]

