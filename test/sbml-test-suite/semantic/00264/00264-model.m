(* 

category:      Test
synopsis:      Basic two reactions with three species in one 1D compartment,
               with one species having a stoichiometry of 2.
componentTags: Compartment, Species, Reaction, Parameter 
testTags:      Amount, NonUnityStoichiometry
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called "compartment".  There are three
species named S1, S2 and S3 and two parameters named k1 and k2.
Compartment "compartment" is 1-dimensional.  The model contains two
reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| 2S1 + S2 -> S3 | $k1 * S1 * S1 * S2 * compartment$  |
| S3 -> 2S1 + S2 | $k2 * S3 * compartment$        |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1                |$1.0 \x 10^-3$ |mole                      |
|Initial amount of S2                |$0.5 \x 10^-3$ |mole                      |
|Initial amount of S3                |$1.0 \x 10^-3$ |mole                      |
|Value of parameter k1               |$ 1.6 \x 10^3$ |metre mole^-1^ second^-1^ |
|Value of parameter k2               |$          0.7$ |second^-1^ |
|Length of compartment "compartment" |$1$             |metre                        |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00264" ];

addCompartment[ compartment, size -> 1, spatialDimensions -> 1];
addSpecies[ S1, initialAmount -> 1.0 10^-3];
addSpecies[ S2, initialAmount -> 0.5 10^-3];
addSpecies[ S3, initialAmount -> 1.0 10^-3];
addParameter[ k1, value -> 1.6 10^3 ];
addParameter[ k2, value -> 0.7 ];
addReaction[ 2S1 + S2 -> S3, reversible -> False,
	     kineticLaw -> k1 * S1 * S1 * S2 * compartment ];
addReaction[ S3 -> 2S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 * compartment ];

makemodel[]

