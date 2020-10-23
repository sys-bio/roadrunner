(* 

category:      Test
synopsis:      Basic two reactions with three species in one 1D compartment 
               and one functionDefinition.
componentTags: Compartment, Species, Reaction, Parameter, FunctionDefinition 
testTags:      Amount, NonUnityCompartment
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called "compartment".  There are three
species named S1, S2 and S3 and two parameters named k1 and k2.
Compartment "compartment" is 1-dimensional.  The model contains two
reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 | $multiply(k1, S1, S2) * compartment$  |
| S3 -> S1 + S2 | $k2 * S3 * compartment$  |]


The model contains one functionDefinition defined as:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | multiply | x, y, z | $x * y * z$ |]


The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1                |$ 1.0 \x 10^-6$ |mole                      |
|Initial amount of S2                |$ 2.0 \x 10^-6$ |mole                      |
|Initial amount of S3                |$ 1.0 \x 10^-6$ |mole                      |
|Value of parameter k1               |$         3.24$ |metre mole^-1^ second^-1^ |
|Value of parameter k2               |$        0.125$ |second^-1^ |
|Length of compartment "compartment" |$          6.8$ |metre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00283" ];

addFunction[ multiply, arguments -> {x, y, z}, math -> x * y * z];
addCompartment[ compartment, spatialDimensions-> 1, size -> 6.8 ];
addSpecies[ S1, initialAmount -> 1.0 10^-6];
addSpecies[ S2, initialAmount -> 2.0 10^-6];
addSpecies[ S3, initialAmount -> 1.0 10^-6];
addParameter[ k1, value -> 3.24 ];
addParameter[ k2, value -> 0.125 ];
addReaction[ S1 + S2 -> S3, reversible -> False,
	     kineticLaw -> multiply[k1, S1, S2] * compartment ];
addReaction[ S3 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 * compartment ];

makemodel[]

