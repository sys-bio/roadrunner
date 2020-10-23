(* 

category:      Test
synopsis:      Basic reactions with four species in a 2-dimensional
               non-unity compartment.
componentTags: Compartment, Species, Reaction, Parameter 
testTags:      Amount, NonUnityCompartment
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called "compartment".  There are four
species named S1, S2, S3 and S4 and two parameters named k1 and k2.
Compartment "compartment" is 2-dimensional.  The model contains two
reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 + S4 | $k1 * S1 * S2 * compartment$  |
| S3 + S4 -> S1 + S2 | $k2 * S3 * S4 * compartment$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1              |$ 1.0 \x 10^-6$ |mole                      |
|Initial amount of S2              |$ 1.5 \x 10^-6$ |mole                      |
|Initial amount of S3              |$ 2.0 \x 10^-6$ |mole                      |
|Initial amount of S4              |$ 1.0 \x 10^-6$ |mole                      |
|Value of parameter k1             |$ 0.75 \x 10^7$ |metre^2^ mole^-1^ second^-1^ |
|Value of parameter k2             |$ 0.25 \x 10^5$ |metre^2^ mole^-1^ second^-1^ |
|Area of compartment "compartment" |$         0.01$ |metre^2^                  |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00204" ];

addCompartment[ compartment, spatialDimensions-> 2, size -> 0.01 ];
addSpecies[ S1, initialAmount -> 1.0 10^-6];
addSpecies[ S2, initialAmount -> 1.5 10^-6];
addSpecies[ S3, initialAmount -> 2.0 10^-6];
addSpecies[ S4, initialAmount -> 1.0 10^-6];
addParameter[ k1, value -> 0.75 10^7 ];
addParameter[ k2, value -> 0.25 10^5 ];
addReaction[ S1 + S2 -> S3 + S4, reversible -> False,
	     kineticLaw -> k1 * S1 * S2 * compartment ];
addReaction[ S3 + S4 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 * S4 * compartment ];

makemodel[]

