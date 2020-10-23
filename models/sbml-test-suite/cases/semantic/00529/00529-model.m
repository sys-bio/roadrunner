(* 

category:      Test
synopsis:      Basic single forward reaction with two species in one
compartment using initialAssignment to set the initial value of one species.
componentTags: Compartment, Species, Reaction, Parameter, InitialAssignment, FunctionDefinition 
testTags:      Amount, InitialValueReassigned
testType:      TimeCourse
levels:        2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

The model contains one compartment called C.  There are two
species called S1 and S2 and two parameters called k1 and k2.  The model
contains one reaction defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $C * k2 * S$  |]

The model contains one initialAssignment:

[{width:30em,margin: 1em auto}| Variable | Formula |
 | S1 | $multiply(k1, S2)$  |]

The initialAssignment uses a functionDefinition defined as:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 |                             multiply | x, y          | $x * y$     |]

Note: SBML's InitialAssignment construct override any declared initial
values.  In this case the initial value declared for species S1 is
inconsistent with the value returned by the InitialAssignment.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*         |*Units*  |
|Initial amount of S1                |$            1$ |mole                      |
|Initial amount of S2                |$1.5 \x 10^-3$ |mole                      |
|Value of parameter k1               |$         0.75$ |dimensionless             |
|Value of parameter k2               |$           5$ |second^-1^                |
|Volume of compartment C |$             $ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

Note: The test data for this model was generated from an analytical solution
of the system of equations.

*)

newcase[ "00529" ];

addFunction[ multiply, arguments -> {x, y}, math -> x * y];
addCompartment[ C, size -> 1];
addSpecies[ S1, initialAmount->1 ];
addSpecies[ S2, initialAmount -> 1.5 10^-3 ];
addParameter[ k1, value -> 0.75 ];
addParameter[ k2, value -> 5 ];
addInitialAssignment[ S1, math -> multiply[k1, S2]];
addReaction[ S1 -> S2, reversible -> False,
	     kineticLaw -> C * k2 * S1];

makemodel[]

