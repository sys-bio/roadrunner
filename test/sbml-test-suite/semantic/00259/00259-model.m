(* 

category:      Test
synopsis:      Basic two reactions with five species in a 0 dimensional compartment 
               where one species is constant.
componentTags: Compartment, Species, Reaction, Parameter 
testTags:      Amount, ConstantSpecies, 0D-Compartment
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called "compartment".  There are five
species named S1, S2, S3, S4 and S5 and two parameters named k1 and k2.
Compartment "compartment" is 0-dimensional.  Species S5 is labeled as
constant and therefore cannot be changed by rules or reactions.  The model
contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 + S4 | $k1 * S1 * S2$  |
| S3 + S4 -> S1 + S2 | $k2 * S3 * S4 / S5$  |]


The initial conditions are as follows:

[{width:30em,margin: 1em auto}|      |*Value*          |*Units*  |
|Initial amount of S1  |$ 1.0 \x 10^-6$ |mole                      |
|Initial amount of S2  |$ 1.5 \x 10^-6$ |mole                      |
|Initial amount of S3  |$ 2.0 \x 10^-6$ |mole                      |
|Initial amount of S4  |$ 0.5 \x 10^-6$ |mole                      |
|Initial amount of S5  |$ 1.0 \x 10^-6$ |mole                      |
|Value of parameter k1 |$  1.3 \x 10^6$ |mole^-1^ second^-1^ |
|Value of parameter k2 |$          0.3$ |mole^-1^ second^-1^ |]

In this example the compartment has its spatialDimensions attribute set to
zero; i.e., it represents a point and therefore cannot have size or units.
The species values must be treated as amounts and not concentrations.

*)

newcase[ "00259" ];

addCompartment[ compartment, spatialDimensions -> 0 ];
addSpecies[ S1, initialAmount -> 1.0 10^-6];
addSpecies[ S2, initialAmount -> 1.5 10^-6];
addSpecies[ S3, initialAmount -> 2.0 10^-6];
addSpecies[ S4, initialAmount -> 0.5 10^-6];
addSpecies[ S5, initialAmount -> 1.0 10^-6, constant->True];
addParameter[ k1, value -> 1.3 10^6 ];
addParameter[ k2, value -> 0.3 ];
addReaction[ S1 + S2 -> S3 + S4, reversible -> False,
	     kineticLaw -> k1 * S1 * S2 ];
addReaction[ S3 + S4 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 * S4/S5 ];

makemodel[]

