(* 

category:      Test
synopsis:      Algebraic rule used to determine rate of species change.
componentTags: Compartment, Species, Reaction, Parameter, AlgebraicRule 
testTags:      Amount, 0D-Compartment
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are four
species called X0, X1, T and S1 and three parameters called k1, k2 and k3.
Compartment C is 0-dimensional.  The model contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| X0 -> T     | $k1 * X0$  |
| T -> X1     | $k2 * S1$  |]

The model contains one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Algebraic |   n/a    | $(1 + k3) * S1 - T$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value* |*Units*  |
|Initial amount of X0          |$1$     |mole                      |
|Initial amount of X1          |$0$     |mole                      |
|Initial amount of T           |$0$     |mole                      |
|Initial amount of S1          |$0$     |mole                      |
|Value of parameter k1         |$0.1$   |second^-1^            |
|Value of parameter k2         |$0.375$ |second^-1^                |
|Value of parameter k3         |$2.5$   |dimensionless                |]

In this example the compartment has its spatialDimensions attribute set to
zero; i.e., it represents a point and therefore cannot have size or units.
The species values must be treated as amounts and not concentrations.

*)

newcase[ "00543" ];

addCompartment[ C, spatialDimensions -> 0 ];
addSpecies[ X0, initialAmount -> 1];
addSpecies[ X1, initialAmount -> 0];
addSpecies[ T, initialAmount -> 0];
addSpecies[ S1, initialAmount -> 0];
addParameter[ k1, value -> 0.1 ];
addParameter[ k2, value -> 0.375 ];
addParameter[ k3, value -> 2.5 ];
addRule[ type->AlgebraicRule, math -> (1+k3) * S1 - T];
addReaction[ X0 -> T, reversible -> False,
	     kineticLaw -> k1 * X0 ];
addReaction[ T -> X1, reversible -> False,
	     kineticLaw -> k2 *S1 ];

makemodel[]

