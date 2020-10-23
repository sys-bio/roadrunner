(* 

category:      Test
synopsis:      Algebraic rule used to determine rate of species change.
componentTags: Compartment, Species, Reaction, Parameter, AlgebraicRule, AssignmentRule 
testTags:      Amount, LocalParameters, InitialValueReassigned
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are four
species called X0, X1, T and S1 and two parameters called k1 and k3.  The model
contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| X0 -> T     | $C * k * X0$  |
| T -> X1     | $C * k * S1$  |]

Reaction T -> X1 defines one local parameter k which has a
scope local to the defining reaction and is different from the global parameter k
used in reaction X0 -> T.

The model contains two rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Algebraic |   n/a    | $(1 + k3) * S1 - T$  |
 | assignmentRule | k | $0.1$  |]

The assignment rule assigns a value to parameter k which has no value
attributed to it by the model definition.  The calculated value should be used.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value* |*Units*  |
|Initial amount of X0          |$1$     |mole                      |
|Initial amount of X1          |$0$     |mole                      |
|Initial amount of T           |$0$     |mole                      |
|Initial amount of S1          |$0$     |mole                      |
|Value of parameter k          |$undeclared$     |second^-1^            |
|Value of parameter k3         |$2.5$   |dimensionless                |
|Value of local parameter k    |$0.375$ |second^-1^                |
|Volume of compartment C       |$1$     |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00705" ];

addCompartment[ C, size -> 1 ];
addSpecies[ X0, initialAmount -> 1];
addSpecies[ X1, initialAmount -> 0];
addSpecies[ T, initialAmount -> 0];
addSpecies[ S1, initialAmount -> 0];
addParameter[ k, constant -> False  ];
addParameter[ k3, value -> 2.5 ];
addRule[ type -> assignmentRule, variable -> k, math -> 0.1];
addRule[ type->AlgebraicRule, math -> (1+k3) * S1 - T];
addReaction[ X0 -> T, reversible -> False,
	     kineticLaw -> C * k * X0 ];
addReaction[ T -> X1, reversible -> False,
	     kineticLaw -> C * k *S1 , parameters -> {k -> 0.375}];

makemodel[]

