(* 

category:      Test
synopsis:      Basic two reactions with three species in a compartment using 
initialAssignment to set the initial value of one species.
componentTags: InitialAssignment, Compartment, Species, Reaction, Parameter 
testTags:      InitialValueReassigned, Amount, AssignedConstantStoichiometry, NonUnityStoichiometry, InitialValueReassigned
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Numeric

Note:  This test is the L3 version of model 520.

The model contains one compartment called C.  There are three species called 
S1, S2 and S3 and four parameters called k1, k2, p1 and p2.  The model contains 
two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> (4 * p2)S3 | $k1 * S1 * S2 * C$  |
| S3 -> S1 + S2 | $k2 * S3 * C$  |]

The model contains one initialAssignment:

[{width:30em,margin: 1em auto}| Variable | Formula |
 | S1 | $p1 * 2$  |]

Note: SBML's InitialAssignment construct override any declared initial
values.  In this case the initial value declared for species S1 is
consistent with the value returned by the InitialAssignment.

The initial conditions are as follows:

[{width 30em,margin: 1em auto}| |*Value*        |*Units*  |
|Initial amount of S1        |$0.25 \x 10^-4$  |mole                      |
|Initial amount of S2        |$2.0 \x 10^-4$  |mole                      |
|Initial amount of S3        |$1.0 \x 10^-4$  |mole                      |
|Value of parameter k1       |$0.75$           |litre mole^-1^ second^-1^ |
|Value of parameter k2       |$0.25$           |second^-1^                |
|Value of parameter p1       |$1.25 \x 10^-5$ |mole                |
|Value of parameter p2       |$0.5$            |dimensionless            |
|Volume of compartment C     |$1$              |litre                  |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) they must be treated as concentrations where they appear in
expressions.

*)

newcase[ "00520" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialAmount -> 0.25 10^-4];
addSpecies[ S2, initialAmount -> 2.0 10^-4];
addSpecies[ S3, initialAmount -> 1.0 10^-4];
addParameter[ k1, value -> 0.75 ];
addParameter[ k2, value -> 0.25 ];
addParameter[ p1, value -> 0.125 10^-4 ];
addParameter[ p2, value -> 0.5 ];
addInitialAssignment[ S1, math -> p1*2];
addReaction[reactants->{S1, S2}, products->{S3}, productStoichiometry->{4 * p1}, 
       reversible -> False,
	     kineticLaw -> k1 * S1 * S2 * C ];
addReaction[ S3 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 * C ];

makemodel[]

