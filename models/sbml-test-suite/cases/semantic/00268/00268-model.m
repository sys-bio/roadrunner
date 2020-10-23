(* 

category:      Test
synopsis:      Basic two reactions with four species in one 0D compartment,
               with non-unity stoichiometry.
componentTags: Compartment, Species, Reaction, Parameter 
testTags:      Amount, NonUnityStoichiometry, 0D-Compartment
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called "compartment".  There are four
species named S1, S2, S3 and S4 and two parameters named k1 and k2.
Compartment "compartment" is 0-dimensional.  The model contains two
reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 + 2S4 | $k1 * S1 * S2$  |
| S3 + S4 -> S1 + S2  | $k2 * S3 * S4$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|      |*Value*          |*Units*  |
|Initial amount of S1  |$ 1.0 \x 10^-6$ |mole                      |
|Initial amount of S2  |$ 0.5 \x 10^-6$ |mole                      |
|Initial amount of S3  |$ 2.0 \x 10^-6$ |mole                      |
|Initial amount of S4  |$            0$ |mole                      |
|Value of parameter k1 |$  0.9 \x 10^6$ |mole^-1^ second^-1^ |
|Value of parameter k2 |$ 0.15 \x 10^6$ |mole^-1^ second^-1^ |]

In this example the compartment has its spatialDimensions attribute set to
zero; i.e., it represents a point and therefore cannot have size or units.
The species values must be treated as amounts and not concentrations.

*)

newcase[ "00268" ];

addCompartment[ compartment, spatialDimensions -> 0 ];
addSpecies[ S1, initialAmount -> 1.0 10^-6];
addSpecies[ S2, initialAmount -> 0.5 10^-6];
addSpecies[ S3, initialAmount -> 2.0 10^-6];
addSpecies[ S4, initialAmount -> 0];
addParameter[ k1, value -> 0.9 10^6 ];
addParameter[ k2, value -> 0.15 10^6 ];
addReaction[ S1 + S2 -> S3 + 2S4, reversible -> False,
	     kineticLaw -> k1 * S1 * S2];
addReaction[ S3 + S4 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 * S4 ];

makemodel[]

