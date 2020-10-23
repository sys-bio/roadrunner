(* 

category:      Test
synopsis:      Basic single forward reaction with three species in one
compartment using an assignmentRule to vary one species.
componentTags: InitialAssignment, Compartment, Species, Reaction, Parameter, AssignmentRule 
testTags:      InitialValueReassigned, Amount, AssignedConstantStoichiometry, NonUnityStoichiometry, InitialValueReassigned
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Numeric

Note:  This test is the L3 version of model 609.

The model contains one compartment called C.  There are three
species called S1, S2 and S3 and three parameters called k1, k2 and p1.  The model
contains one reaction defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> (4 * p1)S2 | $C * k2 * S$  |]

The model contains one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Assignment | S3 | $1 * S2$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1                |$1.5 \x 10^-4$ |mole                      |
|Initial amount of S2                |$1.5 \x 10^-4$ |mole                      |
|Initial amount of S3                |$            1$ |mole                      |
|Value of parameter k1               |$         0.75$ |dimensionless |
|Value of parameter k2               |$           50$ |second^-1^ |
|Value of parameter p1               |$          0.5$ |dimensionless |
|Volume of compartment C             |$            1$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00609" ];

addCompartment[ C, size -> 1];
addSpecies[ S1, initialAmount->1.5 10^-4 ];
addSpecies[ S2, initialAmount -> 1.5 10^-4 ];
addSpecies[ S3, initialAmount -> 1];
addParameter[ k1, value -> 0.75 ];
addParameter[ k2, value -> 5 ];
addParameter[ p1, value -> 0.5 ];
addRule[ type->AssignmentRule, variable -> S3, math ->k1 * S2];
addReaction[reactants->{S1}, products->{S2}, productStoichiometry->{4 * p1}, 
       reversible -> False,
	     kineticLaw -> C * k2 * S1];

makemodel[]

