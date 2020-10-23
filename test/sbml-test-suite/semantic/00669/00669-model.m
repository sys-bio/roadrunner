(* 

category:      Test
synopsis:      Two reactions with five species in one
compartment using an assignmentRules with functionDefinitions to vary two species.
componentTags: Compartment, Species, Reaction, Parameter, FunctionDefinition, AssignmentRule 
testTags:      Amount, NonUnityCompartment
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are five
species called S1, S2, S3, S4 and S5 and three parameters called k1, k2 and
p1.  The model contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S3 | $k1 * S1 * C$  |
| S3 -> S2 | $k2 * S5 * C$  |]

The model contains two rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Assignment | S4 | $func1(S3, p1)$  |
 | Assignment | S5 | $multiply(S4, p1)$  |]

The model contains two functionDefinitions defined as:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | multiply | x, y | $x * y$ |
 | func1 | x, y | $x/(1 + y)$ |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1                |$          1.0$ |mole                      |
|Initial amount of S2                |$            0$ |mole                      |
|Initial amount of S3                |$            0$ |mole                      |
|Initial amount of S4                |$            0$ |mole                      |
|Initial amount of S5                |$            0$ |mole                      |
|Value of parameter k1               |$          0.1$ |second^-1^ |
|Value of parameter k2               |$         0.15$ |second^-1^ |
|Value of parameter p1               |$          2.5$ |dimensionless |
|Volume of compartment "compartment" |$            2.5$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00669" ];

addFunction[ multiply, arguments -> {x, y}, math -> x * y];
addFunction[ func1, arguments -> {x, y}, math -> x/(1 + y)];
addCompartment[ C, size -> 2.5];
addSpecies[ S1, initialAmount -> 1.0];
addSpecies[ S2, initialAmount -> 0];
addSpecies[ S3, initialAmount -> 0];
addSpecies[ S4, initialAmount -> 0];
addSpecies[ S5, initialAmount -> 0];
addParameter[ k1, value -> .1];
addParameter[ k2, value -> .15];
addParameter[ p1, value -> 2.5];
addRule[ type->AssignmentRule, variable -> S4, math ->func1[S3, p1]];
addRule[ type->AssignmentRule, variable -> S5, math ->multiply[S4, p1]];
addReaction[ S1 -> S3, reversible -> False,
	     kineticLaw -> k1 * S1 * C ];
addReaction[ S3 -> S2, modifiers-> S5, reversible -> False,
	     kineticLaw -> k2 * S5 * C ];

makemodel[]

