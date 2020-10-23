(* 

category:      Test
synopsis:      Three reactions with four species in one compartment, 
with an algebraic rule used to determine value of a parameter.
componentTags: Compartment, Species, Reaction, Parameter, AlgebraicRule, EventNoDelay, FunctionDefinition  
testTags:      Amount, InitialValueReassigned
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are four
species called S1, S2, S3 and S4 and three parameters called k1, k2 and k3.
The model contains three reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 | $k1 * S1 * S2 * C$  |
| S3 -> S1 + S2 | $k2 * S3 * C$     |
| S3 -> S1 + S4 | $k3 * S3 * C$     |]

The model contains one rule which must be used to determine the value of
parameter k2:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Algebraic |   n/a    | $subtract(k2, 0.9)$  |]

The model contains one event that assigns value to species S4 defined as:

[{width:30em,margin: 1em auto}| | *Trigger*    | *Delay* | *Assignments* |
 | Event1 | $greaterthan(S4, S3)$ | $-$   | $S4 = 2 * S4$    |]

Both the rule and the event use functionDefinitions defined as:
[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | greaterthan | x, y      | $x > y$   |
 | subtract | x, y      | $x - y$   |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1                |$2.0 \x 10^-1$  |mole                       |
|Initial amount of S2                |$2.0 \x 10^-1$  |mole                       |
|Initial amount of S3                |$0$              |mole                       |
|Initial amount of S4                |$0$              |mole                       |
|Value of parameter k1               |$1.0 \x 10^1$   |litre mole^-1^ second^-1^  |
|Value of parameter k2               |$undeclared$     |second^-1^                 |
|Value of parameter k3               |$0.7$            |second^-1^                 |
|Volume of compartment C             |$1$              |litre                      |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00762" ];

addFunction[ greaterthan, arguments -> {x, y}, math -> x > y];
addFunction[ subtract, arguments -> {x, y}, math -> x - y];
addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialAmount -> 2.0 10^-1];
addSpecies[ S2, initialAmount -> 2.0 10^-1];
addSpecies[ S3, initialAmount -> 0];
addSpecies[ S4, initialAmount -> 0];
addParameter[ k1, value -> 1.0 10^1 ];
addParameter[ k2, constant -> False ];
addParameter[ k3, value -> 0.7 ];
addRule[ type->AlgebraicRule, math -> subtract[k2, 0.9]];
addReaction[ S1 + S2 -> S3, reversible -> False,
	     kineticLaw -> k1 * S1 * S2 * C ];
addReaction[ S3 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 * C ];
addReaction[ S3 -> S1 + S4, reversible -> False,
	     kineticLaw -> k3 * S3 * C ];
addEvent[ trigger -> greaterthan[S4,S3], eventAssignment -> S4 -> 2*S4 ];

makemodel[]

