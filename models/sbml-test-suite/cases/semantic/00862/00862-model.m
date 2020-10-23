(* 

category:      Test
synopsis:      Three reactions with four species in one compartment
using csymbol time within a math expression.
componentTags: Compartment, Species, Reaction, Parameter, CSymbolTime
testTags:      Concentration, NonUnityCompartment
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are four
species called S1, S2, S3 and S4 and three parameters called k1, k2 and k3.
The model contains three reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 | $k1 * S1 * S2 * C * s$  |
| S3 -> S1 + S2 | $k2 * S3 * C * s$     |
| S3 -> S1 + S4 | $k3 * S3 * C * s$     |]
where the symbol 's' denotes the current simulation time.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial concentration of S1                |$2.0$  |mole litre^-1^                       |
|Initial concentration of S2                |$2.0$  |mole litre^-1^                       |
|Initial concentration of S3                |$0$              |mole litre^-1^                       |
|Initial concentration of S4                |$0$              |mole litre^-1^                       |
|Value of parameter k1               |$1.0$   |litre mole^-1^ second^-2^  |
|Value of parameter k2               |$0.9$            |second^-2^                 |
|Value of parameter k3               |$0.7$            |second^-2^                 |
|Volume of compartment C |$0.002$              |litre                      |]

*)

newcase[ "00862" ];

addCompartment[ C, size -> 0.002 ];
addSpecies[ S1, initialConcentration -> 2.0];
addSpecies[ S2, initialConcentration -> 2.0];
addSpecies[ S3, initialConcentration -> 0];
addSpecies[ S4, initialConcentration -> 0];
addParameter[ k1, value -> 1.0 ];
addParameter[ k2, value -> 0.9 ];
addParameter[ k3, value -> 0.7 ];
addReaction[ S1 + S2 -> S3, reversible -> False,
	     kineticLaw -> k1 * S1 * S2 * C *\[LeftAngleBracket]s, "time"\[RightAngleBracket]];
addReaction[ S3 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 * C *\[LeftAngleBracket]s, "time"\[RightAngleBracket]];
addReaction[ S3 -> S1 + S4, reversible -> False,
	     kineticLaw -> k3 * S3 * C *\[LeftAngleBracket]s, "time"\[RightAngleBracket]];

makemodel[]

