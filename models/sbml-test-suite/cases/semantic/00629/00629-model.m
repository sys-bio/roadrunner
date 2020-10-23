(* 

category:      Test
synopsis:      Algebraic rule used to determine rate of species change.
componentTags: Compartment, Species, Reaction, Parameter, AlgebraicRule, AssignmentRule, FunctionDefinition 
testTags:      Amount, InitialValueReassigned
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are four
species called X0, X1, T and S1 and five parameters called k1, k2 
k3, p1 and p2.  The model contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| X0 -> T     | $C * k1 * X0$  |
| T -> X1     | $C * k2 * S1$  |]

The model contains two rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Algebraic |   n/a    | $(1 + k3) * S1 - T$  |
 | assignmentRule | k1 | $multiply(p1, p2)$  |]

The assignmentRule applies the functionDefinition defined as:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | multiply | x, y      | $x * y$   |]
 
The assignment rule assigns a value to parameter k1 which is inconsistent with the value
attributed to the parameter by the model definition.  The calculated value should be used.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value* |*Units*  |
|Initial amount of X0          |$1$     |mole                      |
|Initial amount of X1          |$0$     |mole                      |
|Initial amount of T           |$0$     |mole                      |
|Initial amount of S1          |$0$     |mole                      |
|Value of parameter k1         |$2$     |second^-1^            |
|Value of parameter k2         |$0.375$ |second^-1^                |
|Value of parameter k3         |$2.5$   |dimensionless                |
|Value of parameter p1         |$10$    |dimensionless                |
|Value of parameter p2         |$0.01$  |second^-1^                |
|Volume of compartment C       |$1$     |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00629" ];

addFunction[ multiply, arguments -> {x, y}, math -> x * y];
addCompartment[ C, size -> 1 ];
addSpecies[ X0, initialAmount -> 1];
addSpecies[ X1, initialAmount -> 0];
addSpecies[ T, initialAmount -> 0];
addSpecies[ S1, initialAmount -> 0];
addParameter[ k1, value -> 2, constant -> False  ];
addParameter[ k2, value -> 0.375 ];
addParameter[ k3, value -> 2.5 ];
addParameter[ p1, value -> 10 ];
addParameter[ p2, value -> 0.01 ];
addRule[ type -> assignmentRule, variable -> k1, math -> multiply[p1, p2]];
addRule[ type->AlgebraicRule, math -> (1+k3) * S1 - T];
addReaction[ X0 -> T, reversible -> False,
	     kineticLaw -> C * k1 * X0 ];
addReaction[ T -> X1, reversible -> False,
	     kineticLaw -> C * k2 *S1 ];

makemodel[]

