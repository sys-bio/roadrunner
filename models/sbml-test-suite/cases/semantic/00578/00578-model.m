(* 

category:      Test
synopsis:      Basic two reactions with three species in one compartment,
               and two species labeled as a boundary condition
componentTags: Compartment, Species, Reaction, Parameter 
testTags:      Concentration, BoundaryCondition
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

The model contains one compartment called C.  There are three
species called S1, S2 and S3 and two parameters called k1 and k2.  Species S1
and S2 are labeled as SBML boundary species.  The model contains two
reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 | $k1 * S1 * S2 * C$  |
| S3 -> S1 + S2 | $k2 * S3 * C$     |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value*          |*Units*                   |
|Initial concentration of S1   |$1.0 \x 10^-2$  |mole litre^-1^                      |
|Initial concentration of S2   |$2.0 \x 10^-2$  |mole litre^-1^                      |
|Initial concentration of S3   |$1.0 \x 10^-2$  |mole litre^-1^                      |
|Value of parameter k1         |$0.75$           |litre mole^-1^ second^-1^ |
|Value of parameter k2         |$0.25$           |second^-1^                |
|Volume of compartment C       |$1$              |litre                     |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

newcase[ "00578" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialConcentration -> 1.0 10^-2, boundaryCondition -> True];
addSpecies[ S2, initialConcentration -> 2.0 10^-2, boundaryCondition -> True];
addSpecies[ S3, initialConcentration -> 1.0 10^-2];
addParameter[ k1, value -> 0.75 ];
addParameter[ k2, value -> 0.25 ];
addReaction[ S1 + S2 -> S3, reversible -> False,
	     kineticLaw -> k1 * S1 * S2 * C ];
addReaction[ S3 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 * C ];

makemodel[]

