(* 

category:      Test
synopsis:      Linear chain of reactions in one compartment.
componentTags: Compartment, Species, Reaction, Parameter 
testTags:      Concentration
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

The model contains one compartment called C.  There are four
species called S1, S2, S3 and S4 and three parameters called k1, k2 and k3.
The model contains three reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $k1 * S1 * C$  |
| S2 -> S3 | $k2 * S2 * C$  |
| S3 -> S4 | $k3 * S3 * C$  |] 

The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value*          |*Units*  |
|Initial concentration of S1   |$1.0 \x 10^-1$  |mole litre^-1^           |
|Initial concentration of S2   |$0$              |mole litre^-1^           |
|Initial concentration of S3   |$0$              |mole litre^-1^           |
|Initial concentration of S4   |$0$              |mole litre^-1^           |
|Value of parameter k1         |$0.7$            |second^-1^     |
|Value of parameter k2         |$0.5$            |second^-1^     |
|Value of parameter k3         |$1$              |second^-1^     |
|Volume of compartment C       |$1$              |litre          |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

newcase[ "00585" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialConcentration -> 1.0 10^-1];
addSpecies[ S2, initialConcentration -> 0];
addSpecies[ S3, initialConcentration -> 0];
addSpecies[ S4, initialConcentration -> 0];
addParameter[ k1, value -> 0.7 ];
addParameter[ k2, value -> 0.5 ];
addParameter[ k3, value -> 1 ];
addReaction[ S1 -> S2, reversible -> False,
	     kineticLaw -> k1 * S1 * C ];
addReaction[ S2 -> S3, reversible -> False,
	     kineticLaw -> k2 * S2 * C ];
addReaction[ S3 -> S4, reversible -> False,
	     kineticLaw -> k3 * S3 * C ];

makemodel[]

