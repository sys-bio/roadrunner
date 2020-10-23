(* 

category:      Test
synopsis:      Basic two reactions with three species and parameters both global
               and local to reactions.
componentTags: Compartment, Species, Reaction, Parameter 
testTags:      Concentration, LocalParameters
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are three
species called S1, S2 and S3 and one global parameter called k.  The model
contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $k * S1 * C$  |
| S2 -> S3 | $k * S2 * C$  |]

Reaction S2 -> S3 defines one local parameter k.

Note that the id of the local parameter k shadows the global parameter k.
Within the defining reaction, the value of the locally-defined parameter k
must be used.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value*          |*Units*  |
|Initial concentration of S1   |$3.0 \x 10^-1$ |mole litre^-1^                      |
|Initial concentration of S2   |$            0$ |mole litre^-1^                      |
|Initial concentration of S3   |$            0$ |mole litre^-1^                      |
|Value of parameter k          |$            1$ |second^-1^ |
|Value of local parameter k    |$            2$ |second^-1^ |
|Volume of compartment C       |$            1$ |litre                     |]

*)

newcase[ "00596" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialConcentration -> 3.0 10^-1];
addSpecies[ S2, initialConcentration -> 0];
addSpecies[ S3, initialConcentration -> 0];
addParameter[ k, value -> 1];
addReaction[ S1 -> S2, reversible -> False,
	     kineticLaw -> k * S1 * C];
addReaction[ S2 -> S3, reversible -> False,
	     kineticLaw -> k * S2 * C, parameters -> {k -> 2} ];

makemodel[]

