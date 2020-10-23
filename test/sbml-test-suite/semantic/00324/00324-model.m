(* 

category:      Test
synopsis:      Two reactions and a rate rule with four species in a 0D compartment. 
componentTags: Compartment, Species, Reaction, Parameter, RateRule
testTags:      Amount, 0D-Compartment
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called "compartment".  There are four
species named S1, S2, S3 and S4 and two parameters named k1 and k2.
Compartment "compartment" is 0-dimensional.  The model contains two
reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 | $k1 * S1 * S2$  |
| S3 -> S1 + S2 | $k2 * S3$  |]

The model contains one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Rate | S4 | $1 10^-2$  |]


The initial conditions are as follows:

[{width:30em,margin: 1em auto}|      |*Value*          |*Units*  |
|Initial amount of S1  |$1.5 \x 10^-2$ |mole                      |
|Initial amount of S2  |$2.0 \x 10^-2$ |mole                      |
|Initial amount of S3  |$4.0 \x 10^-2$ |mole                      |
|Initial amount of S4  |$  1 \x 10^-2$ |mole                      |
|Value of parameter k1 |$0.69 \x 10^2$ |mole^-1^ second^-1^ |
|Value of parameter k2 |$0.23 \x 10^-1$ |second^-1^ |]

In this example the compartment has its spatialDimensions attribute set to
zero; i.e., it represents a point and therefore cannot have size or units.
The species values must be treated as amounts and not concentrations.

*)

newcase[ "00324" ];

addCompartment[ compartment, spatialDimensions -> 0 ];
addSpecies[ S1, initialAmount -> 1.5 10^-2 ];
addSpecies[ S2, initialAmount -> 2.0 10^-2];
addSpecies[ S3, initialAmount -> 4.0 10^-2];
addSpecies[ S4, initialAmount -> 1 10^-2 ];
addParameter[ k1, value -> .69 10^2];
addParameter[ k2, value -> .23 10^-1];
addRule[ type->RateRule, variable -> S4, math -> 1 10^-2];
addReaction[ S1 + S2 -> S3, reversible -> False,
	     kineticLaw -> k1 * S1 * S2 ];
addReaction[ S3 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 ];

makemodel[]

