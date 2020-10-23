(* 

category:      Test
synopsis:      Basic two reactions with four species in a compartment 
               of non-unity volume. 
componentTags: Compartment, Species, Reaction, Parameter
testTags:      Concentration, NonUnityCompartment
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are four
species called S1, S2, S3 and S4 and two parameters called k1 and k2.  The
model contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 + S4 | $k1 * S1 * S2 * C$  |
| S3 + S4 -> S1 + S2 | $k2 * S3 * S4 * C$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value*          |*Units*  |
|Initial concentration of S1   |$          1.5$ |mole litre^-1^                      |
|Initial concentration of S2   |$          1.7$ |mole litre^-1^                      |
|Initial concentration of S3   |$          2.0$ |mole litre^-1^                      |
|Initial concentration of S4   |$          1.0$ |mole litre^-1^                      |
|Value of parameter k1         |$         0.75$ |litre mole^-1^ second^-1^ |
|Value of parameter k2         |$         0.25$ |litre mole^-1^ second^-1^ |
|Volume of compartment C       |$          0.3$ |litre                     |]

*)

newcase[ "00603" ];

addCompartment[ C, size -> 0.3 ];
addSpecies[ S1, initialConcentration -> 1.5 ];
addSpecies[ S2, initialConcentration -> 1.7 ];
addSpecies[ S3, initialConcentration -> 2.0 ];
addSpecies[ S4, initialConcentration -> 1.0 ];
addParameter[ k1, value -> .75  ];
addParameter[ k2, value -> .25  ];
addReaction[ S1 + S2 -> S3 + S4, reversible -> False,
	     kineticLaw -> k1 * S1 * S2 * C ];
addReaction[ S3 + S4 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 * S4 * C ];

makemodel[]

