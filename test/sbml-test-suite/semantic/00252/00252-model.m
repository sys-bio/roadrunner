(* 

category:      Test
synopsis:      Basic reactions with four species in a 2 dimensional compartment 
               where one species is constant.
componentTags: Compartment, Species, Reaction, Parameter 
testTags:      Amount, ConstantSpecies
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called "compartment".  There are four
species named S1, S2, S3 and S4 and two parameters named k1 and k2.
Compartment "compartment" is 2-dimensional.  Species S4 is labeled as
constant and therefore cannot be changed by rules or reactions.  The model
contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 | $k1 * S1 * S2 * compartment$  |
| S3 -> S1 + S2 | $k2 * S3 * S4 * compartment$  |]


The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1              |$ 1.0 \x 10^-3$ |mole                      |
|Initial amount of S2              |$ 1.5 \x 10^-3$ |mole                      |
|Initial amount of S3              |$0.75 \x 10^-3$ |mole                      |
|Initial amount of S4              |$1.25 \x 10^-3$ |mole                      |
|Value of parameter k1             |$ 1.68 \x 10^3$ |metre^2^ mole^-1^ second^-1^ |
|Value of parameter k2             |$ 0.27 \x 10^3$ |metre^2^ mole^-1^ second^-1^ |
|Area of compartment "compartment" |$            1$ |metre^2^                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00252" ];

addCompartment[ compartment, size -> 1, spatialDimensions -> 2 ];
addSpecies[ S1, initialAmount -> 1.0 10^-3];
addSpecies[ S2, initialAmount -> 1.5 10^-3 ];
addSpecies[ S3, initialAmount -> 0.75 10^-3 ];
addSpecies[ S4, initialAmount -> 1.25 10^-3 , constant->True];
addParameter[ k1, value -> 1.68 10^3  ];
addParameter[ k2, value -> 0.27 10^3 ];
addReaction[ S1 + S2 -> S3, reversible -> False,
	     kineticLaw -> k1 * S1 * S2 * compartment ];
addReaction[ S3 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 * S4 * compartment ];

makemodel[]

