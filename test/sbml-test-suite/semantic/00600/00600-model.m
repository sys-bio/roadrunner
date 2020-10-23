(* 

category:      Test
synopsis:      Basic two reactions with five species in a compartment 
               where one species is constant.
componentTags: Compartment, Species, Reaction, Parameter 
testTags:      Concentration, ConstantSpecies
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are five
species called S1, S2, S3, S4 and S5 and two parameters called k1 and k2.
Species S5 is labeled as constant and therefore cannot be changed by rules
or reactions.  The model contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 + S4 | $k1 * S1 * S2 * C$  |
| S3 + S4 -> S1 + S2 | $k2 * S3 * S4 * C / S5$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value*         |*Units*  |
|Initial concentration of S1   |$ 1.0 \x 10^-6$ |mole litre^-1^                      |
|Initial concentration of S2   |$ 1.5 \x 10^-6$ |mole litre^-1^                      |
|Initial concentration of S3   |$ 2.0 \x 10^-6$ |mole litre^-1^                      |
|Initial concentration of S4   |$ 0.5 \x 10^-6$ |mole litre^-1^                      |
|Initial concentration of S5   |$ 1.0 \x 10^-6$ |mole litre^-1^                      |
|Value of parameter k1         |$  1.3 \x 10^6$ |litre mole^-1^ second^-1^ |
|Value of parameter k2         |$          0.3$ |litre mole^-1^ second^-1^ |
|Volume of compartment C       |$            1$ |litre                     |]

*)

newcase[ "00600" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialConcentration -> 1.0 10^-6];
addSpecies[ S2, initialConcentration -> 1.5 10^-6];
addSpecies[ S3, initialConcentration -> 2.0 10^-6];
addSpecies[ S4, initialConcentration -> 0.5 10^-6];
addSpecies[ S5, initialConcentration -> 1.0 10^-6, constant->True];
addParameter[ k1, value -> 1.3 10^6 ];
addParameter[ k2, value -> 0.3 ];
addReaction[ S1 + S2 -> S3 + S4, reversible -> False,
	     kineticLaw -> k1 * S1 * S2 * C ];
addReaction[ S3 + S4 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 * S4 * C/S5 ];

makemodel[]

