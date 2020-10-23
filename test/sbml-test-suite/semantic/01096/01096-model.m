(* 

category:      Test
synopsis:      Single reversible reaction.
componentTags: InitialAssignment, Compartment, Species, Reaction, Parameter 
testTags:      InitialValueReassigned, Amount, AssignedConstantStoichiometry, NonUnityStoichiometry
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Numeric

Note:  This test is the L3 version of model 827.

The model contains one compartment called C.  There are two species called 
S1 and S2 and three parameters called kf, kr and p1.  The model contains  one reaction
defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 <-> (4 * p1)S2 | $(kf * S1 - kr * S2) * C$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value*       |*Units*  |
|Initial amount of S1        |$1.0$ |mole                      |
|Initial amount of S2        |$0$ |mole                      |
|Value of parameter kf       |$0.9$          |second^-1^ |
|Value of parameter kr       |$0.075$          |second^-1^ |
|Value of parameter p1       |$0.5$          |dimensionless |
|Volume of compartment C     |$1$             |litre                  |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00827" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.0];
addSpecies[ S2, initialAmount -> 0];
addParameter[ kf, value -> 0.9 ];
addParameter[ kr, value -> 0.075 ];
addParameter[ p1, value -> 0.5 ];
addReaction[ reactants->{S1}, products->{S2}, productStoichiometry->{4 * p1},
	     kineticLaw -> (kf * S1 - kr * S2) * C ];

makemodel[]

