(* 

category:      Test
synopsis:      Basic two reactions with four species in a compartment 
               of non-unity volume. 
componentTags: Compartment, Species, Reaction, Parameter
testTags:      Amount, NonUnityCompartment
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called "compartment".  There are four
species named S1, S2, S3 and S4 and two parameters named k1 and k2.  The
model contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 + S4 | $k1 * S1 * S2 * compartment$  |
| S3 + S4 -> S1 + S2 | $k2 * S3 * S4 * compartment$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1                |$          1.5$ |mole                      |
|Initial amount of S2                |$          1.7$ |mole                      |
|Initial amount of S3                |$          2.0$ |mole                      |
|Initial amount of S4                |$          1.0$ |mole                      |
|Value of parameter k1               |$         0.75$ |litre mole^-1^ second^-1^ |
|Value of parameter k2               |$         0.25$ |litre mole^-1^ second^-1^ |
|Volume of compartment "compartment" |$          0.3$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00077" ];

addCompartment[ compartment, size -> 0.3 ];
addSpecies[ S1, initialAmount -> 1.5 ];
addSpecies[ S2, initialAmount -> 1.7 ];
addSpecies[ S3, initialAmount -> 2.0 ];
addSpecies[ S4, initialAmount -> 1.0 ];
addParameter[ k1, value -> .75  ];
addParameter[ k2, value -> .25  ];
addReaction[ S1 + S2 -> S3 + S4, reversible -> False,
	     kineticLaw -> k1 * S1 * S2 * compartment ];
addReaction[ S3 + S4 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 * S4 * compartment ];

makemodel[]

