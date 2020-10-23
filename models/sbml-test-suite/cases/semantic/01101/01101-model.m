(* 

category:      Test
synopsis:      Three reactions with four species in one compartment
using csymbol time within a math expression.
componentTags: Compartment, Species, Reaction, Parameter, CSymbolTime, InitialAssignment
testTags:      InitialValueReassigned, Amount, LocalParameters, AssignedConstantStoichiometry, NonUnityStoichiometry
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Numeric

Note:  This test is the L3 version of model 900.

The model contains one compartment called C.  There are four
species called S1, S2, S3 and S4 and two parameters called k1 and p1.
The model contains three reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| (2 * p1)S1 + S2 -> S3 | $k1 * S1 * S2 * C * s$  |
| S3 -> S1 + S2 | $k1 * S3 * C * s$     |
| S3 -> S1 + S4 | $k1 * S3 * C * s$     |]
where the symbol 's' denotes the current simulation time.

Both reactions S3 -> S1 + S2 and S3 -> S1 + S4 define a local parameters k1 which has a
scope local to the defining reaction and are different from each other and the global parameter k1
used in the reaction S1 + S2 -> S3.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1                |$2.0$  |mole                       |
|Initial amount of S2                |$2.0$  |mole                       |
|Initial amount of S3                |$0$              |mole                       |
|Initial amount of S4                |$0$              |mole                       |
|Value of parameter k1               |$1.0$   |litre mole^-1^ second^-2^  |
|Value of parameter p1               |$1.0$   |dimensionless  |
|Value of local parameter k1 (R2)               |$0.9$            |second^-2^                 |
|Value of local parameter k1 (R3)               |$0.7$            |second^-2^                 |
|Volume of compartment C |$1$              |litre                      |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00900" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialAmount -> 2.0];
addSpecies[ S2, initialAmount -> 2.0];
addSpecies[ S3, initialAmount -> 0];
addSpecies[ S4, initialAmount -> 0];
addParameter[ k1, value -> 1.0 ];
addParameter[ p1, value -> 1.0 ];
addReaction[ reactants->{S1, S2}, products->{S3},
             reactantStoichiometry->{2 * p1, 1}, reversible -> False, 
	     kineticLaw -> k1 * S1 * S2 * C *\[LeftAngleBracket]s, "time"\[RightAngleBracket]];
addReaction[ S3 -> S1 + S2, reversible -> False,
	     kineticLaw -> k1 * S3 * C *\[LeftAngleBracket]s, "time"\[RightAngleBracket],
	     parameters -> {k1 -> 0.9}];
addReaction[ S3 -> S1 + S4, reversible -> False,
	     kineticLaw -> k1 * S3 * C *\[LeftAngleBracket]s, "time"\[RightAngleBracket],
	     parameters -> {k1 -> 0.7}];

makemodel[]

