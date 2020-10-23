(* 

category:      Test
synopsis:      Basic two reactions with four species in one 2D compartment,
               with non-unity stoichiometry.
componentTags: Compartment, Species, Reaction, Parameter 
testTags:      Amount, NonUnityStoichiometry
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called "compartment".  There are four
species named S1, S2, S3 and S4 and two parameters named k1 and k2.
Compartment "compartment" is 2-dimensional.  The model contains two
reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 + 2S4 | $k1 * S1 * S2 * compartment$  |
| S3 + S4 -> S1 + S2  | $k2 * S3 * S4 * compartment$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1              |$ 1.0 \x 10^-6$ |mole                      |
|Initial amount of S2              |$ 0.5 \x 10^-6$ |mole                      |
|Initial amount of S3              |$ 2.0 \x 10^-6$ |mole                      |
|Initial amount of S4              |$            0$ |mole                      |
|Value of parameter k1             |$  0.9 \x 10^6$ |metre^2^ mole^-1^ second^-1^ |
|Value of parameter k2             |$ 0.15 \x 10^6$ |metre^2^ mole^-1^ second^-1^ |
|Area of compartment "compartment" |$1$             |metre^2^                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00266" ];

addCompartment[ compartment, size -> 1, spatialDimensions -> 2 ];
addSpecies[ S1, initialAmount -> 1.0 10^-6];
addSpecies[ S2, initialAmount -> 0.5 10^-6];
addSpecies[ S3, initialAmount -> 2.0 10^-6];
addSpecies[ S4, initialAmount -> 0];
addParameter[ k1, value -> 0.9 10^6 ];
addParameter[ k2, value -> 0.15 10^6 ];
addReaction[ S1 + S2 -> S3 + 2S4, reversible -> False,
	     kineticLaw -> k1 * S1 * S2 * compartment ];
addReaction[ S3 + S4 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 * S4 * compartment ];

makemodel[]

