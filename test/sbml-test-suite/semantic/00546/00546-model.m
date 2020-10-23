(* 

category:      Test
synopsis:      Basic two reactions with four species in one 0D compartment with
an algebraic rule used to determine rate of change of one species.
componentTags: Compartment, Species, Reaction, Parameter, AlgebraicRule 
testTags:      Amount, 0D-Compartment
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are four
species called S1, S2, S3 and S4 and two parameters called k1 and k2.
Compartment C is 0-dimensional.  The model contains two reactions defined as:


[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 | $k1 * S1 * S2$  |
| S3 -> S1 + S2 | $k2 * S3$     |]

The model contains one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Algebraic |   n/a    | $S4 - S1$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*                   |
|Initial amount of S1                |$1.0 \x 10^-4$  |mole                      |
|Initial amount of S2                |$2.0 \x 10^-4$  |mole                      |
|Initial amount of S3                |$1.0 \x 10^-4$  |mole                      |
|Initial amount of S4                |$1.0 \x 10^-4$  |mole                      |
|Value of parameter k1               |$0.75$           |mole^-1^ second^-1^ |
|Value of parameter k2               |$0.25$           |second^-1^                |]

In this example the compartment has its spatialDimensions attribute set to
zero; i.e., it represents a point and therefore cannot have size or units.
The species values must be treated as amounts and not concentrations.

*)

newcase[ "00546" ];

addCompartment[ C, spatialDimensions -> 0 ];
addSpecies[ S1, initialAmount -> 1.0 10^-4];
addSpecies[ S2, initialAmount -> 2.0 10^-4];
addSpecies[ S3, initialAmount -> 1.0 10^-4];
addSpecies[ S4, initialAmount -> 1.0 10^-4];
addParameter[ k1, value -> 0.75 ];
addParameter[ k2, value -> 0.25 ];
addRule[ type->AlgebraicRule, math -> S4 - S1];
addReaction[ S1 + S2 -> S3, reversible -> False,
	     kineticLaw -> k1 * S1 * S2 ];
addReaction[ S3 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 ];

makemodel[]

