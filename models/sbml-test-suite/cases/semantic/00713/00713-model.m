(* 

category:      Test
synopsis:      Rate rule using a functionDefinition used to determine value of parameter
               which is used in a reaction.
componentTags: Compartment, Species, Reaction, Parameter, FunctionDefinition, RateRule 
testTags:      Amount, NonConstantParameter, NonUnityCompartment
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are four
species called S1, S2, S3 and S4; three constant parameters called k2, k3 and
p1 and one varying parameter called k1.  The model contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 + S4 | $k1 * S1 * S2 * C$  |
| S3 + S4 -> S1 + S2 | $k2 * S3 * S4 * C$  |]

The model contains one rule that determines the value of parameter k1:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Rate | k1 | $add(k2, k3) * p1$  |]


The model contains one functionDefinition defined as:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | add | x, y | $x + y$ |]


The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*         |*Units*  |
|Initial amount of S1                |$ 1.0 \x 10^-6$ |mole                      |
|Initial amount of S2                |$ 1.5 \x 10^-6$ |mole                      |
|Initial amount of S3                |$ 2.0 \x 10^-6$ |mole                      |
|Initial amount of S4                |$ 0.5 \x 10^-6$ |mole                      |
|Value of parameter k1               |$    1 \x 10^6$ |litre mole^-1^ second^-1^ |
|Value of parameter k2               |$  0.3 \x 10^6$ |litre mole^-1^ second^-1^ |
|Value of parameter k3               |$  0.7 \x 10^6$ |litre mole^-1^ second^-1^ |
|Value of parameter p1               |$            1$ |second^-1^                |
|Volume of compartment C |$            0.15$ |litre                     |]
The species values are given as amounts of substance to make it easier to

use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00713" ];

addFunction[ add, arguments -> {x, y}, math -> x + y];
addCompartment[ C, size -> 0.15 ];
addSpecies[ S1, initialAmount -> 1.0 10^-6];
addSpecies[ S2, initialAmount -> 1.5 10^-6];
addSpecies[ S3, initialAmount -> 2.0 10^-6];
addSpecies[ S4, initialAmount -> 0.5 10^-6];
addParameter[ k1, value -> 1 10^6, constant->False ];
addParameter[ k2, value -> 0.3 10^6 ];
addParameter[ k3, value -> 0.7 10^6 ];
addParameter[ p1, value -> 1 ];
addRule[ type->RateRule, variable -> k1, math -> p1*add[k2, k3]];
addReaction[ S1 + S2 -> S3 + S4, reversible -> False,
	     kineticLaw -> k1 * S1 * S2 * C ];
addReaction[ S3 + S4 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 * S4 * C ];

makemodel[]

