(* 

category:      Test
synopsis:      Basic single forward reaction with two species in one
compartment using initialAssignment to set the initial value of one parameter.
componentTags: Compartment, Species, Reaction, Parameter, InitialAssignment 
testTags:      Amount, InitialValueReassigned
testType:      TimeCourse
levels:        2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

The model contains one compartment called "compartment".  There are two
species named S1 and S2 and two parameters named k1 and k2.  The model
contains one reaction defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $compartment * k1 * S1$  |]

The model contains one initialAssignment:

[{width:30em,margin: 1em auto}| Variable | Formula |
 | k1 | $k2 / 100$  |]

Note: InitialAssignments override any declared initial values.  In this
case the value calculated by the initialAssignment is consistent with the
value declared.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*         |*Units*  |
|Initial amount of S1                |$            1$ |mole                      |
|Initial amount of S2                |$          1.5$ |mole                      |
|Value of parameter k1               |$          0.5$ |second^-1^ |
|Value of parameter k2               |$           50$ |second^-1^ |
|Volume of compartment "compartment" |$             $ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

Note: The test data for this model was generated from an analytical solution
of the system of equations.

*)

newcase[ "00136" ];

addCompartment[ compartment, size -> 1];
addSpecies[ S1, initialAmount -> 1 ];
addSpecies[ S2, initialAmount -> 1.5 ];
addParameter[ k1, value -> .5 ];
addParameter[ k2, value -> 50 ];
addInitialAssignment[ k1, math -> k2/100];
addReaction[ S1 -> S2, reversible -> False,
	     kineticLaw -> compartment * k1 * S1];

makemodel[]

