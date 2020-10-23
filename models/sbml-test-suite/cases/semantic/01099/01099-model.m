(* 

category:      Test
synopsis:      Basic single forward reaction with two species in one compartment
using csymbol time within a math expression.
componentTags: InitialAssignment, Compartment, Species, Reaction, Parameter, CSymbolTime
testTags:      InitialValueReassigned, Amount, AssignedConstantStoichiometry, NonUnityStoichiometry
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Analytic

Note:  This test is the L3 version of model 898.

The model contains one compartment called C.  There are two
species called S1 and S2 and two parameters called k1 and p1.  The model contains
one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> (2 * p1) S2 | $k1 * S1 * C * t$ |]
where the symbol 't' denotes the current simulation time.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*        |
|Initial amount of S1                |$1.5 \x 10^-3$  |mole           |
|Initial amount of S2                |$0$              |mole           |
|Value of parameter k1               |$1$              |second^-2^     |
|Value of parameter p1               |$1$              |dimensionless     |
|Volume of compartment C |$1$              |litre          |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but (as
per usual SBML principles) their symbols represent their values in
concentration units where they appear in expressions.

Note: The test data for this model was generated from an analytical solution
of the system of equations.

*)

newcase[ "00898" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.5 10^-3 ];
addSpecies[ S2, initialAmount -> 0 ];
addParameter[ k1, value -> 1 ];
addParameter[ p1, value -> 1 ];
addReaction[ reactants->{S1}, products->{S2}, 
             productStoichiometry->{2 * p1}, reversible -> False, 
 kineticLaw -> k1 * S1 * C *\[LeftAngleBracket]t, "time"\[RightAngleBracket]];

makemodel[]

