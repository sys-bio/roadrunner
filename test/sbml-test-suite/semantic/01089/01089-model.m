(* 

category:      Test
synopsis:      Two reactions and a rate rule with four species in a compartment. 
componentTags: InitialAssignment, Compartment, Species, Reaction, Parameter, RateRule
testTags:      InitialValueReassigned, Amount, AssignedConstantStoichiometry, NonUnityStoichiometry
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Numeric

Note:  This test is the L3 version of model 725.

The model contains one compartment called C.  There are four
species called S1, S2, S3 and S4 and three parameters called k1, k2 and p1.  The
model contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 | $k1 * S1 * S2 * C$  |
| (4 * p1) S3 -> S1 + S2 | $k2 * S3 * C$  |]

The model contains one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Rate | S4 | $1 10^-2$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1                |$1.5 \x 10^-2$ |mole                      |
|Initial amount of S2                |$2.0 \x 10^-2$ |mole                      |
|Initial amount of S3                |$1.5 \x 10^-2$ |mole                      |
|Initial amount of S4                |$  1 \x 10^-2$ |mole                      |
|Value of parameter k1               |$0.75 \x 10^2$ |litre mole^-1^ second^-1^ |
|Value of parameter k2               |$0.25 \x 10^-1$ |second^-1^ |
|Value of parameter p1               |$0.25$ |dimensionless |
|Volume of compartment C |$            1$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00725" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.5 10^-2 ];
addSpecies[ S2, initialAmount -> 2.0 10^-2];
addSpecies[ S3, initialAmount -> 1.5 10^-2];
addSpecies[ S4, initialAmount -> 1 10^-2 ];
addParameter[ k1, value -> .75 10^2];
addParameter[ k2, value -> .25 10^-1];
addParameter[ p1, value -> 0.25];
addRule[ type->RateRule, variable -> S4, math -> 1 10^-2];
addReaction[ S1 + S2 -> S3, reversible -> False,
	     kineticLaw -> k1 * S1 * S2 * C ];
addReaction[reactants->{S3}, products->{S1, S2}, reactantStoichiometry->{4 * p1}, 
       reversible -> False,
	     kineticLaw -> k2 * S3 * C ];

makemodel[]

