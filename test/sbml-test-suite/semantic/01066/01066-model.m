(* 

category:      Test
synopsis:      Basic reactions with three species in a compartment 
               involving a stoichiometryMath element.
componentTags: InitialAssignment, Compartment, Species, Reaction, Parameter 
testTags:      InitialValueReassigned, Amount, AssignedConstantStoichiometry, NonUnityStoichiometry
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Numeric

Note:  This test is the L3 version of model 69.

The model contains one compartment called "compartment".  There are three
species named S1, S2 and S3 and three parameters named k1, k2 and p1.  The
model contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3         | $k1 * S1 * S2 * compartment$  |
| (2 * p1)S3 -> S1  +  S2 | $k2 * S3 * compartment$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*         |*Units*  |
|Initial amount of S1                |$1.0 \x 10^-1$ |mole                      |
|Initial amount of S2                |$2.0 \x 10^-1$ |mole                      |
|Initial amount of S3                |$1.0 \x 10^-1$ |mole                      |
|Value of parameter k1               |$ 1.7 \x 10^1$ |litre mole^-1^ second^-1^ |
|Value of parameter k2               |$          0.3$ |second^-1^ |
|Value of parameter p1               |$            1$ |dimensionless |
|Volume of compartment "compartment" |$            1$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00069" ];

addCompartment[ compartment, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.0 10^-1];
addSpecies[ S2, initialAmount -> 2.0 10^-1];
addSpecies[ S3, initialAmount -> 1.0 10^-1];
addParameter[ k1, value -> 1.7 10^1 ];
addParameter[ k2, value -> 0.3 ];
addParameter[ p1, value -> 1 ];
addReaction[ S1 + S2 -> S3, reversible -> False,
	     kineticLaw -> k1 * S1 * S2 * compartment ];
addReaction[ reactants->{S3}, products->{S1, S2}, reactantStoichiometry->{2 * p1},
       reversible -> False,
	     kineticLaw -> k2 * S3 * compartment ];

makemodel[]

