(* 

category:      Test
synopsis:      Two reactions with three species in one compartment, 
with an algebraic rule used to determine value of a parameter.
componentTags: Compartment, Species, Reaction, Parameter, AlgebraicRule 
testTags:      Amount, InitialValueReassigned
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are three
species called X0, X1 and T and two parameters called k1 and k2.  The model
contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| X0 -> T     | $C * k1 * X0$  |
| T -> X1     | $C * k2 * T$  |]

The model contains one rule which must be used to determine the value of
parameter k1:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Algebraic |   n/a    | $k1 - 0.1$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value* |*Units*  |
|Initial amount of X0          |$1$     |mole                      |
|Initial amount of X1          |$0$     |mole                      |
|Initial amount of T           |$0$     |mole                      |
|Value of parameter k1         |$undeclared$   |second^-1^            |
|Value of parameter k2         |$0.375$ |second^-1^                |
|Volume of compartment C       |$1$     |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00536" ];

addCompartment[ C, size -> 1 ];
addSpecies[ X0, initialAmount -> 1];
addSpecies[ X1, initialAmount -> 0];
addSpecies[ T, initialAmount -> 0];
addParameter[ k1, constant -> False ];
addParameter[ k2, value -> 0.375 ];
addRule[ type->AlgebraicRule, math -> k1 - 0.1];
addReaction[ X0 -> T, reversible -> False,
	     kineticLaw -> C * k1 * X0 ];
addReaction[ T -> X1, reversible -> False,
	     kineticLaw -> C * k2 *T ];

makemodel[]

