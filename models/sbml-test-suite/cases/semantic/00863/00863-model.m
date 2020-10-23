(* 

category:      Test
synopsis:      Basic single forward reaction with two species in one compartment
using csymbol time within a math expression.
componentTags: Compartment, Species, Reaction, Parameter, CSymbolTime 
testTags:      Amount, BoundaryCondition
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

The model contains one compartment called C.  There are two
species called S1 and S2 and one parameter called k1.  Species S1
is labeled as an SBML boundary species.  The model contains
one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $k1 * S1 * C * t$ |]
where the symbol 't' denotes the current simulation time.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*        |
|Initial amount of S1                |$1.5 \x 10^-1$  |mole           |
|Initial amount of S2                |$0$              |mole           |
|Value of parameter k1               |$1$              |second^-2^     |
|Volume of compartment C |$1$              |litre          |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but (as
per usual SBML principles) their symbols represent their values in
concentration units where they appear in expressions.

Note: The test data for this model was generated from an analytical solution
of the system of equations.

*)

newcase[ "00863" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.5 10^-1 , boundaryCondition -> True];
addSpecies[ S2, initialAmount -> 0 ];
addParameter[ k1, value -> 1 ];
addReaction[ S1 -> S2, reversible -> False, 
 kineticLaw -> k1 * S1 * C *\[LeftAngleBracket]t, "time"\[RightAngleBracket]];

makemodel[]

