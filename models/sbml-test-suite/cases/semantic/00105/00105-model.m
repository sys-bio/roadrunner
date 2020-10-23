(* 

category:      Test
synopsis:      Basic two reactions with three species in one varying compartment 
               and one functionDefinition
componentTags: Compartment, Species, Reaction, Parameter, FunctionDefinition, RateRule 
testTags:      Amount, NonConstantCompartment, NonUnityCompartment
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one varying compartment called C.  There are three
species called S1, S2 and S3 and three parameters called k1, k2 and p1.  The
model contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 | $multiply(k1, S1, S2) * C$  |
| S3 -> S1 + S2 | $k2 * S3 * C$  |]

The model contains one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Rate | C | $-p1 * C$  |]


The model contains one functionDefinition defined as:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | multiply | x, y, z | $x * y * z$ |]


The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1                |$          1.0$ |mole                      |
|Initial amount of S2                |$          2.0$ |mole                      |
|Initial amount of S3                |$          1.0$ |mole                      |
|Value of parameter k1               |$         0.75$ |litre mole^-1^ second^-1^ |
|Value of parameter k2               |$         0.25$ |second^-1^ |
|Value of parameter p1               |$          0.1$ |second^-1^ |
|Volume of compartment C |$            1$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00105" ];

addFunction[ multiply, arguments -> {x, y, z}, math -> x * y * z];
addCompartment[ C, constant -> False, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.0];
addSpecies[ S2, initialAmount -> 2.0];
addSpecies[ S3, initialAmount -> 1.0];
addParameter[ k1, value -> 0.75 ];
addParameter[ k2, value -> 0.25 ];
addParameter[ p1, value -> 0.1 ];
addRule[ type->RateRule, variable -> C, math -> -p1 * C];
addReaction[ S1 + S2 -> S3, reversible -> False,
	     kineticLaw -> multiply[k1, S1, S2] * C ];
addReaction[ S3 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 * C ];

makemodel[]

