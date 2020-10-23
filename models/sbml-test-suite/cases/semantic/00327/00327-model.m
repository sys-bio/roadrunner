(* 

category:      Test
synopsis:      One reactions and two rate rules with four species in a 0D compartment. 
componentTags: Compartment, Species, Reaction, Parameter, RateRule 
testTags:      Amount, 0D-Compartment
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

The model contains one compartment called "compartment".  There are four
species named S1, S2, S3 and S4 and two parameters named k1 and k2.
Compartment "compartment" is 0-dimensional.  The model contains one
reaction defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $k1 * S1$  |]

The model contains two rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Rate | S3 | $k1  *  0.5$  |
 | Rate | S4 | $-k2  *  0.5$  |]


The initial conditions are as follows:

[{width:30em,margin: 1em auto}|      |*Value*          |*Units*  |
|Initial amount of S1  |$          1.5$ |mole                      |
|Initial amount of S2  |$          2.0$ |mole                      |
|Initial amount of S3  |$          1.5$ |mole                      |
|Initial amount of S4  |$            4$ |mole                      |
|Value of parameter k1 |$         0.75$ |second^-1^ |
|Value of parameter k2 |$         0.25$ |second^-1^ |]

In this example the compartment has its spatialDimensions attribute set to
zero; i.e., it represents a point and therefore cannot have size or units.
The species values must be treated as amounts and not concentrations.

Note: The test data for this model was generated from an analytical solution
of the system of equations.

*)

newcase[ "00327" ];

addCompartment[ compartment, spatialDimensions -> 0 ];
addSpecies[ S1, initialAmount -> 1.5 ];
addSpecies[ S2, initialAmount -> 2.0];
addSpecies[ S3, initialAmount -> 1.5];
addSpecies[ S4, initialAmount -> 4];
addParameter[ k1, value -> .75 ];
addParameter[ k2, value -> .25 ];
addRule[ type->RateRule, variable -> S3, math -> k1  *  0.5];
addRule[ type->RateRule, variable -> S4, math -> -k2  *  0.5];
addReaction[ S1 -> S2, reversible -> False,
	     kineticLaw -> k1 * S1 ];

makemodel[]

