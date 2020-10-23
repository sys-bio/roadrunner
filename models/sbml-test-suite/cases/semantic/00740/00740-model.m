(* 

category:      Test
synopsis:      Rate rule using a functionDefinition used to determine value of parameter
               which is used in a reaction.
componentTags: Compartment, Species, Reaction, Parameter, FunctionDefinition, RateRule 
testTags:      Concentration, NonConstantParameter, NonUnityCompartment
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are two
species called S1 and S2; two constant parameters called k2 and k3 and one varying
parameter called k1.  The
model contains one reaction defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $k1 * S1 * C$  |]

The model contains one rule that determines the value of parameter k1:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Rate | k1 | $add(k2, k3)$  |]


The model contains one functionDefinition defined as:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | add | x, y | $x + y$ |]


The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*         |*Units*  |
|Initial concentration of S1                |$1.5 \x 10^-3$ |mole litre^-1^                      |
|Initial concentration of S2                |$            0$ |mole litre^-1^                      |
|Value of parameter k1               |$            1$ |second^-1^ |
|Value of parameter k2               |$          0.2$ |second^-1^ |
|Value of parameter k3               |$          0.3$ |lsecond^-1^ |
|Volume of compartment C |$            2.5$ |litre                     |]

*)

newcase[ "00740" ];

addFunction[ add, arguments -> {x, y}, math -> x + y];
addCompartment[ C, size -> 2.5 ];
addSpecies[ S1, initialConcentration -> 1.5 10^-3 ];
addSpecies[ S2, initialConcentration -> 0 ];
addParameter[ k1, value -> 1, constant->False ];
addParameter[ k2, value -> 0.2];
addParameter[ k3, value -> 0.3];
addRule[ type->RateRule, variable -> k1, math -> add[k2,k3]];
addReaction[ S1 -> S2, reversible -> False,
	     kineticLaw -> k1 * S1 * C ];

makemodel[]

