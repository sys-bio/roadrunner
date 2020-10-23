(* 

category:      Test
synopsis:      Linear chain of reactions in one compartment 
using csymbol time within a math expression.
componentTags: Compartment, Species, Reaction, Parameter, CSymbolTime
testTags:      Amount, LocalParameters
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are four
species called S1, S2, S3 and S4 and one parameter called k2.
The model contains three reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $k2 * S1 * C * time$  |
| S2 -> S3 | $k2 * S2 * C * time$  |
| S3 -> S4 | $k2 * S3 * C * time$  |] 
where the symbol 'time' denotes the current simulation time.

Both reactions S1 -> S2 and S3 -> S4 define a local parameters k2 which has a
scope local to the defining reaction and are different from each other and the 
global parameter k2 used in the reaction S2 -> S3.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1                |$1.0 \x 10^-4$  |mole           |
|Initial amount of S2                |$0$              |mole           |
|Initial amount of S3                |$0$              |mole           |
|Initial amount of S4                |$0$              |mole           |
|Value of parameter k2               |$0.5$            |second^-2^     |
|Value of local parameter k2 (R1)               |$0.7$            |second^-2^     |
|Value of local parameter k2 (R3)               |$1$              |second^-2^     |
|Volume of compartment C |$1$              |litre          |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00897" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.0 10^-4];
addSpecies[ S2, initialAmount -> 0];
addSpecies[ S3, initialAmount -> 0];
addSpecies[ S4, initialAmount -> 0];
addParameter[ k2, value -> 0.5 ];
addReaction[ S1 -> S2, reversible -> False,
	     kineticLaw -> k2 * S1 * C*\[LeftAngleBracket]time, "time"\[RightAngleBracket],
	     parameters -> {k2 -> 0.7}];
addReaction[ S2 -> S3, reversible -> False,
	     kineticLaw -> k2 * S2 * C*\[LeftAngleBracket]time, "time"\[RightAngleBracket] ];
addReaction[ S3 -> S4, reversible -> False,
	     kineticLaw -> k2 * S3 * C*\[LeftAngleBracket]time, "time"\[RightAngleBracket],
	     parameters -> {k2 -> 1}];

makemodel[]

