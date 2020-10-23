(* 

category:      Test
synopsis:      Algebraic rule used to determine rate of species change.
componentTags: Compartment, Species, Reaction, Parameter, AlgebraicRule 
testTags:      Amount, LocalParameters, InitialValueReassigned
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are four
species called X0, X1, T and S1 and one parameter k.  The model contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| X0 -> T     | $C * k * X0$  |
| T -> X1     | $C * k * S1$  |]

Reaction X0 -> T defines one local parameter k.  Reaction T -> X1 defines another 
local parameter k.  Note that these parameters have a
scope local to the defining reaction.

The model contains one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Algebraic |   n/a    | $(1 + k) * S1 - T$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value* |*Units*  |
|Initial amount of X0          |$1$     |mole                      |
|Initial amount of X1          |$0$     |mole                      |
|Initial amount of T           |$0$     |mole                      |
|Initial amount of S1          |$0$     |mole                      |
|Value of parameter k          |$2.5$   |dimensionless                |
|Value of local parameter k    |$0.1$   |second^-1^            |
|Value of local parameter k    |$0.375$ |second^-1^                |
|Volume of compartment C       |$1$     |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00565" ];

addCompartment[ C, size -> 1 ];
addSpecies[ X0, initialAmount -> 1];
addSpecies[ X1, initialAmount -> 0];
addSpecies[ T, initialAmount -> 0];
addSpecies[ S1, initialAmount -> 0];
addParameter[ k, value -> 2.5 ];
addRule[ type->AlgebraicRule, math -> (1+k) * S1 - T];
addReaction[ X0 -> T, reversible -> False,
	     kineticLaw -> C * k * X0, parameters -> {k -> 0.1}  ];
addReaction[ T -> X1, reversible -> False,
	     kineticLaw -> C * k *S1, parameters -> {k -> 0.375}  ];

makemodel[]

