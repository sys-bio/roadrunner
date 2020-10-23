(* 

category:      Test
synopsis:      Basic single forward reaction with three species in a compartment 
               where one species is constant.
componentTags: Compartment, Species, Reaction, Parameter 
testTags:      Concentration, ConstantSpecies
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

The model contains one compartment called C.  There are three
species called S1, S2 and S3 and one parameter called k1.  Species S3 is
labeled as constant and therefore cannot be changed by rules or reactions.
The model contains one reaction defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $k1 * S1 * S3 * C$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value*         |*Units*  |
|Initial concentration of S1   |$          1.5$ |mole litre^-1^                      |
|Initial concentration of S2   |$            0$ |mole litre^-1^                      |
|Initial concentration of S3   |$            2$ |mole litre^-1^                      |
|Value of parameter k1         |$          1.5$ |litre mole^-1^ second^-1^ |
|Volume of compartment C       |$            1$ |litre                     |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

newcase[ "00598" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialConcentration -> 1.5 ];
addSpecies[ S2, initialConcentration -> 0 ];
addSpecies[ S3, initialConcentration -> 2, constant->True ];
addParameter[ k1, value -> 1.5 ];
addReaction[ S1 -> S2, reversible -> False,
	     kineticLaw -> k1 * S1 * S3 * C ];

makemodel[]

