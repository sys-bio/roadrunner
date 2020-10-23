(* 

category:      Test
synopsis:      Algebraic rule used to determine rate of species change.
componentTags: Compartment, Species, Reaction, Parameter, AlgebraicRule, AssignmentRule 
testTags:      Concentration, NonUnityCompartment, InitialValueReassigned
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are four
species called X0, X1, T and S1 and three parameters called k1, k2 and k3.  The model
contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| X0 -> T     | $C * k1 * X0$  |
| T -> X1     | $C * k2 * S1$  |]

The model contains two rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Algebraic |   n/a    | $(1 + k3) * S1 - T$  |
 | assignmentRule | k1 | $0.1$  |]

The assignment rule assigns a value to parameter k1 which is consistent with the value
attributed to the parameter by the model definition.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value* |*Units*  |
|Initial concentration of X0          |$1$     |mole litre^-1^                      |
|Initial concentration of X1          |$0$     |mole litre^-1^                      |
|Initial concentration of T           |$0$     |mole litre^-1^                      |
|Initial concentration of S1          |$0$     |mole litre^-1^                      |
|Value of parameter k1         |$0.1$   |second^-1^            |
|Value of parameter k2         |$0.375$ |second^-1^                |
|Value of parameter k3         |$2.5$   |dimensionless                |
|Volume of compartment C       |$2.5$     |litre                     |]


*)

newcase[ "00687" ];

addCompartment[ C, size -> 2.5 ];
addSpecies[ X0, initialConcentration -> 1];
addSpecies[ X1, initialConcentration -> 0];
addSpecies[ T, initialConcentration -> 0];
addSpecies[ S1, initialConcentration -> 0];
addParameter[ k1, value -> 0.1, constant -> False  ];
addParameter[ k2, value -> 0.375 ];
addParameter[ k3, value -> 2.5 ];
addRule[ type -> assignmentRule, variable -> k1, math -> 0.1];
addRule[ type->AlgebraicRule, math -> (1+k3) * S1 - T];
addReaction[ X0 -> T, reversible -> False,
	     kineticLaw -> C * k1 * X0 ];
addReaction[ T -> X1, reversible -> False,
	     kineticLaw -> C * k2 *S1 ];

makemodel[]

