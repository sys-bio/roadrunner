(* 

category:      Test
synopsis:      Two reactions with four species in one
compartment using an assignmentRule and a rateRule to vary species.
componentTags: InitialAssignment, Compartment, Species, Reaction, Parameter, RateRule, AssignmentRule 
testTags:      InitialValueReassigned, Amount, AssignedConstantStoichiometry, NonUnityStoichiometry, InitialValueReassigned
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Numeric

Note:  This test is the L3 version of model 728.

The model contains one compartment called C.  There are four
species called S1, S2, S3 and S and four parameters called k1, k2, k3 and p1.
The model contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| (4 * p1)S1 -> S3 | $k1 * S1 * C$  |
| S3 -> S1 | $k2 * S3 * C$  |]

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
|Value of parameter p1               |$0.25$ |dimensionless |
|Volume of compartment C |$            1$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00728" ];

addCompartment[ C, size -> 1];
addSpecies[ S1, initialAmount->1 10^-5 ];
addSpecies[ S2, initialAmount -> 1.5 10^-5];
addSpecies[ S3, initialAmount -> 1 10^-5];
addSpecies[ S4];
addParameter[ k1, value -> .015];
addParameter[ k2, value -> .5 ];
addParameter[ k3, value -> 1.5 ];
addParameter[ p1, value -> 0.25];
addRule[ type->AssignmentRule, variable -> S4, math ->k3 * S1];
addRule[ type->RateRule, variable -> S2, math ->k2 * S3-k1 * S1];
addReaction[reactants->{S1}, products->{S3}, reactantStoichiometry->{4 * p1}, 
       reversible -> False,
	     kineticLaw -> k1 * S1 * C ];
addReaction[ S3 -> S1, reversible -> False,
	     kineticLaw -> k2 * S3 * C ];

makemodel[]

