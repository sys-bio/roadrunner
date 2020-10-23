(* 

category:      Test
synopsis:      Rate rule using a functionDefinition used to determine value of parameter
               which is used in a reaction.
componentTags: Compartment, Species, Reaction, Parameter, FunctionDefinition, RateRule 
testTags:      Concentration, NonConstantParameter, NonUnityCompartment
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are three
species called S1, S2 and S3; four constant parameters called k2, k3, p1 and
p2 and one varying parameter called k1.  The model contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 | $k1 * S1 * S2 * C$  |
| S3 -> S1 + S2 | $k2 * S3 * C$  |]

The model contains one rule that determines the value of parameter k1:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Rate | k1 | $p1* add(p2 * k2, k3)$  |]


The model contains one functionDefinition defined as:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | add | x, y | $x + y$ |]


The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*         |*Units*  |
|Initial concentration of S1                |$          1.0$ |mole litre^-1^                      |
|Initial concentration of S2                |$          2.0$ |mole litre^-1^                      |
|Initial concentration of S3                |$          1.0$ |mole litre^-1^                      |
|Value of parameter k1               |$          1.7$ |litre mole^-1^ second^-1^ |
|Value of parameter k2               |$          0.3$ |second^-1^ |
|Value of parameter k3               |$         -0.1$ |litre mole^-1^ second^-1^ |
|Value of parameter p1               |$            1$ |second^-1^                |
|Value of parameter p2               |$            1$ |litre mole^-1^                 |
|Volume of compartment C |$            1.25$ |litre                     |]

*)

newcase[ "00742" ];

addFunction[ add, arguments -> {x, y}, math -> x + y];
addCompartment[ C, size -> 1.25 ];
addSpecies[ S1, initialConcentration -> 1.0 ];
addSpecies[ S2, initialConcentration -> 2.0 ];
addSpecies[ S3, initialConcentration -> 1.0 ];
addParameter[ k1, value -> 1.7 , constant->False];
addParameter[ k2, value -> 0.3 ];
addParameter[ k3, value -> -0.1 ];
addParameter[ p1, value -> 1 ];
addParameter[ p2, value -> 1 ];
addRule[ type->RateRule, variable -> k1, math -> p1*add[p2*k2, k3]];
addReaction[ S1 + S2 -> S3, reversible -> False,
	     kineticLaw -> k1 * S1 * S2 * C ];
addReaction[ S3 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 * C ];

makemodel[]

