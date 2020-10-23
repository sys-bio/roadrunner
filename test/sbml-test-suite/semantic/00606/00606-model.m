(* 

category:      Test
synopsis:      Basic two reactions with four species in one varying compartment
               and one functionDefinition.
componentTags: Compartment, Species, Reaction, Parameter, FunctionDefinition 
testTags:      Concentration, NonUnityCompartment
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are four
species called S1, S2, S3 and S4 and two parameters called k1 and k2.  The
model contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 + S4 | $multiply(k1,multiply(S1,S2)) * C$  |
| S3 + S4 -> S1 + S2 | $k2 * S3 * S4 * C$  |]


The model contains one functionDefinition defined as:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | multiply | x, y | $x * y$ |]


The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value*          |*Units*  |
|Initial concentration of S1   |$0.5 \x 10^-3$ |mole litre^-1^                      |
|Initial concentration of S2   |$1.0 \x 10^-3$ |mole litre^-1^                      |
|Initial concentration of S3   |$2.0 \x 10^-3$ |mole litre^-1^                      |
|Initial concentration of S4   |$1.5 \x 10^-3$ |mole litre^-1^                      |
|Value of parameter k1         |$0.75 \x 10^2$ |litre mole^-1^ second^-1^ |
|Value of parameter k2         |$0.25 \x 10^3$ |litre mole^-1^ second^-1^ |
|Volume of compartment C       |$         12.5$ |litre                     |]

*)

newcase[ "00606" ];

addFunction[ multiply, arguments -> {x, y}, math -> x * y];
addCompartment[ C, size -> 12.5 ];
addSpecies[ S1, initialConcentration -> 0.5 10^-3];
addSpecies[ S2, initialConcentration -> 1.0 10^-3];
addSpecies[ S3, initialConcentration -> 2.0 10^-3];
addSpecies[ S4, initialConcentration -> 1.5 10^-3];
addParameter[ k1, value -> 0.75 10^2 ];
addParameter[ k2, value -> 0.25 10^3 ];
addReaction[ S1 + S2 -> S3 + S4, reversible -> False,
	     kineticLaw -> multiply[k1,multiply[S1,S2]] * C ];
addReaction[ S3 + S4 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 * S4 * C ];

makemodel[]

