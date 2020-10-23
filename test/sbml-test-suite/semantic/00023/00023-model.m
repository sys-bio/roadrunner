(* 

category:      Test
synopsis:      Three reactions with four species in one compartment, with
               one species labeled constant.
componentTags: Compartment, Species, Reaction, Parameter 
testTags:      Amount, ConstantSpecies, BoundaryCondition
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called "compartment".  There are four
species named S1, S2, S3 and S4 and four parameters named k1, k2, k3 and
k4.  Species S2 is labeled a SBML boundary condition and is also labeled
constant.  The model contains four reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2    | $k1 * S1 * compartment$    |
| S2 -> S1    | $k2 * S2 * compartment$    |
| S2 -> S3 + S4 | $k3 * S2 * compartment$    |
| S3 + S4 -> S2 | $k4 * S3 * S4 * compartment$ |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1                |$1.0 \x 10^-1$  |mole                       |
|Initial amount of S2                |$2.0 \x 10^-1$  |mole                       |
|Initial amount of S3                |$0$              |mole                       |
|Initial amount of S4                |$0$              |mole                       |
|Value of parameter k1               |$0.75$           |second^-1^                 |
|Value of parameter k2               |$0.25$           |second^-1^                 |
|Value of parameter k3               |$0.1$           |second^-1^                 |
|Value of parameter k4               |$0.1$            |litre mole^-1^ second^-1^  |
|Volume of compartment "compartment" |$1$              |litre                      |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00023" ];

addCompartment[ compartment, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.0 10^-1];
addSpecies[ S2, initialAmount -> 2.0 10^-1,constant-> True, 
                                    boundaryCondition->True];
addSpecies[ S3, initialAmount -> 0];
addSpecies[ S4, initialAmount -> 0];
addParameter[ k1, value -> 0.75 ];
addParameter[ k2, value -> 0.25 ];
addParameter[ k3, value -> 0.1 ];
addParameter[ k4, value -> 0.1 ];
addReaction[ S1 -> S2, reversible -> False,
	     kineticLaw -> k1 * S1 * compartment ];
addReaction[ S2 -> S1, reversible -> False,
	     kineticLaw -> k2 * S2 * compartment ];
addReaction[ S2 -> S3 + S4, reversible -> False,
	     kineticLaw -> k3 * S2 * compartment ];
addReaction[ S3 + S4 -> S2, reversible -> False,
	     kineticLaw -> k4 * S3 * S4 * compartment ];



