(* 

category:      Test
synopsis:      Basic two reactions with three species in one 0D compartment,
               with one species having a stoichiometry of 2.
componentTags: Compartment, Species, Reaction, Parameter 
testTags:      Amount, NonUnityStoichiometry, 0D-Compartment
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called "compartment".  There are three
species named S1, S2 and S3 and two parameters named k1 and k2.
Compartment "compartment" is 0-dimensional.  The model contains two
reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| 2S1 + S2 -> S3 | $k1 * S1 * S1 * S2$  |
| S3 -> 2S1 + S2 | $k2 * S3t$        |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|      |*Value*          |*Units*  |
|Initial amount of S1  |$1.0 \x 10^-2$ |mole                      |
|Initial amount of S2  |$0.5 \x 10^-2$ |mole                      |
|Initial amount of S3  |$1.0 \x 10^-2$ |mole                      |
|Value of parameter k1 |$ 1.6 \x 10^2$ |mole^-1^ second^-1^ |
|Value of parameter k2 |$          0.7$ |second^-1^ |]

In this example the compartment has its spatialDimensions attribute set to
zero; i.e., it represents a point and therefore cannot have size or units.
The species values must be treated as amounts and not concentrations.

*)

newcase[ "00265" ];

addCompartment[ compartment, spatialDimensions -> 0];
addSpecies[ S1, initialAmount -> 1.0 10^-2];
addSpecies[ S2, initialAmount -> 0.5 10^-2];
addSpecies[ S3, initialAmount -> 1.0 10^-2];
addParameter[ k1, value -> 1.6 10^2 ];
addParameter[ k2, value -> 0.7 ];
addReaction[ 2S1 + S2 -> S3, reversible -> False,
	     kineticLaw -> k1 * S1 * S1 * S2 ];
addReaction[ S3 -> 2S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 ];

makemodel[]

