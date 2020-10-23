(* 

category:      Test
synopsis:      Basic reaction with two species in a compartment 
               of non-unity volume. 
componentTags: Compartment, Species, Reaction, Parameter 
testTags:      Concentration, NonUnityCompartment
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are three
species called S1, S2 and S3 and two parameters called k1 and k2.  The model
contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 | $k1 * S1 * S2 * C$  |
| S3 -> S1 + S2 | $k2 * S3 * C$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value*          |*Units*  |
|Initial concentration of S1   |$1.0 \x 10^-4$ |mole litre^-1^                      |
|Initial concentration of S2   |$2.0 \x 10^-4$ |mole litre^-1^                      |
|Initial concentration of S3   |$1.5 \x 10^-4$ |mole litre^-1^                      |
|Value of parameter k1         |$0.75 \x 10^4$ |litre mole^-1^ second^-1^ |
|Value of parameter k2         |$0.25 \x 10^-2$ |second^-1^ |
|Volume of compartment C       |$          0.7$ |litre                     |]

*)

newcase[ "00587" ];

addCompartment[ C, size -> 0.7 ];
addSpecies[ S1, initialConcentration -> 1.0 10^-4];
addSpecies[ S2, initialConcentration -> 2.0 10^-4];
addSpecies[ S3, initialConcentration -> 1.5 10^-4];
addParameter[ k1, value -> .75 10^4];
addParameter[ k2, value -> .25 10^-2];
addReaction[ S1 + S2 -> S3, reversible -> False,
	     kineticLaw -> k1 * S1 * S2 * C ];
addReaction[ S3 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 * C ];

makemodel[]

