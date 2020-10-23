(* 

category:      Test
synopsis:      Basic single forward reaction with two species in one compartment
using csymbol time within an initialAssignment.
componentTags: Compartment, Species, Reaction, Parameter, CSymbolTime, InitialAssignment 
testTags:      Amount, InitialValueReassigned
testType:      TimeCourse
levels:        2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

The model contains one compartment called C.  There are two
species called S1 and S2 and two parameters called k1 and p1.  The model contains
one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $k1 * S1 * C$ |]

The model contains one initialAssignment:

[{width:30em,margin: 1em auto}| Variable | Formula |
 | S2       | $p1 * S1 * time$  |]
where the symbol 'time' denotes the current simulation time.

Note: SBML's InitialAssignment take effect at time t < 0 and continue to have effect 
up to and including time t = 0, overriding any initial values.  In this model, the 
initial value for the species S1 has not been explicitly declared and must be 
calculated using the InitialAssignment at time = 0. 


The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*        |
|Initial amount of S1                |$1.5 \x 10^-3$  |mole           |
|Initial amount of S2                |$undeclared$     |mole           |
|Value of parameter p1               |$-0.5$           |second^-1^     |
|Value of parameter k1               |$1$              |second^-1^     |
|Volume of compartment C |$1$              |litre          |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but (as
per usual SBML principles) their symbols represent their values in
concentration units where they appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

newcase[ "00877" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.5 10^-3 ];
addSpecies[ S2 ];
addParameter[ k1, value -> 1 ];
addParameter[ p1, value -> -0.5 ];
addInitialAssignment[ S2, math -> p1 * S1 *\[LeftAngleBracket]time, "time"\[RightAngleBracket]];
addReaction[ S1 -> S2, reversible -> False, 
 kineticLaw -> k1 * S1 * C];

makemodel[]

