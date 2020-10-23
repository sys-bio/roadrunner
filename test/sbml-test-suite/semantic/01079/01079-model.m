(* 

category:      Test
synopsis:      Basic single forward reaction with two species in one
compartment using initialAssignment to set the initial value of one species.
componentTags: InitialAssignment, Compartment, Species, Reaction, Parameter 
testTags:      InitialValueReassigned, Amount, AssignedConstantStoichiometry, NonUnityStoichiometry, InitialValueReassigned
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Analytic

Note:  This test is the L3 version of model 518.

The model contains one compartment called C.  There are two
species called S1 and S2 and three parameters called k1, k2 and p1.  The model contains 
one reaction defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> (4 * p1)S2 | $C * k2 * S$  |]

The model contains one initialAssignment:

[{width:30em,margin: 1em auto}| Variable | Formula |
 | S1       | $k1 * S2$  |]

Note: SBML's InitialAssignment construct override any declared initial
values.  In this case the initial value declared for species S1 is
consistent with the value returned by the InitialAssignment.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*      |*Units*  |
|Initial amount of S1                |$1.125$          |mole                      |
|Initial amount of S2                |$1.5$        |mole                      |
|Value of parameter k1               |$0.75$       |dimensionless             |
|Value of parameter k2               |$5$         |second^-1^                |
|Value of parameter p1               |$0.5$   |dimensionless                 |
|Volume of compartment C |$1$          |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

Note: The test data for this model was generated from an analytical solution
of the system of equations.

*)

newcase[ "00518" ];

addCompartment[ C, size -> 1];
addSpecies[ S1, initialAmount -> 1.125 ];
addSpecies[ S2, initialAmount -> 1.5 ];
addParameter[ k1, value -> 0.75 ];
addParameter[ k2, value -> 5 ];
addParameter[ p1, value -> 0.5 ];
addInitialAssignment[ S1, math -> k1 * S2];
addReaction[reactants->{S1}, products->{S2}, productStoichiometry->{4 * p1}, 
       reversible -> False,
	     kineticLaw -> C * k2 * S1];

makemodel[]

