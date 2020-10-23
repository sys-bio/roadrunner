(* 

category:      Test
synopsis:      Basic single forward reaction with two species in one
compartment using initialAssignment to set the initial value of one species.
componentTags: Compartment, Species, Reaction, Parameter, InitialAssignment 
testTags:      Amount, InitialValueReassigned
testType:      TimeCourse
levels:        2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

The model contains one compartment called "compartment".  There are two
species named S1 and S2 and two parameters named k1 and k2.  The model
contains one reaction defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $compartment * k2 * S$  |]

The model contains one initialAssignment:

[{width:30em,margin: 1em auto}| Variable | Formula |
 | S1       | $k1 * S2$  |]

Note: SBML's InitialAssignment construct override any declared initial
values.  In this model, the initial value for the species S1 has not been
explicitly declared and must be calculated using the InitialAssignment.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*         |*Units*  |
|Initial amount of S1                |$   undeclared$ |mole                      |
|Initial amount of S2                |$1.5 \x 10^-2$ |mole                      |
|Value of parameter k1               |$         0.75$ |dimensionless             |
|Value of parameter k2               |$           50$ |second^-1^                |
|Volume of compartment "compartment" |$            1$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

newcase[ "00036" ];

addCompartment[ compartment, size -> 1];
addSpecies[ S1 ];
addSpecies[ S2, initialAmount -> 1.5 10^-2 ];
addParameter[ k1, value -> 0.75 ];
addParameter[ k2, value -> 50 ];
addInitialAssignment[ S1, math -> k1 * S2];
addReaction[ S1 -> S2, reversible -> False,
	     kineticLaw -> compartment * k2 * S1];

makemodel[]

