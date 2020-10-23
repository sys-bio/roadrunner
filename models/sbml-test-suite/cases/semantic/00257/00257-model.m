(* 

category:      Test
synopsis:      Basic single forward reaction with three species in a 
               0 dimensional compartment, where one species is constant.
componentTags: Compartment, Species, Reaction, Parameter 
testTags:      Amount, ConstantSpecies, 0D-Compartment
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called "compartment".  There are three
species named S1, S2 and S3 and one parameter named k1.  Compartment
"compartment" is 0-dimensional.  Species S3 is labeled as constant and
therefore cannot be changed by rules or reactions.  The model contains one
reaction defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $k1 * S1 * S3$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|      |*Value*          |*Units*  |
|Initial amount of S1  |$          1.5$ |mole                      |
|Initial amount of S2  |$          0.5$ |mole                      |
|Initial amount of S3  |$          1.2$ |mole                      |
|Value of parameter k1 |$         1.78$ |mole^-1^ second^-1^ |]

In this example the compartment has its spatialDimensions attribute set to
zero; i.e., it represents a point and therefore cannot have size or units.
The species values must be treated as amounts and not concentrations.

*)

newcase[ "00257" ];

addCompartment[ compartment, spatialDimensions -> 0 ];
addSpecies[ S1, initialAmount -> 1.5 ];
addSpecies[ S2, initialAmount -> 0.5 ];
addSpecies[ S3, initialAmount -> 1.2, constant->True ];
addParameter[ k1, value -> 1.78 ];
addReaction[ S1 -> S2, reversible -> False,
	     kineticLaw -> k1 * S1 * S3 ];

makemodel[]

