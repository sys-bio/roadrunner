(* 

category:      Test
synopsis:      Basic single forward reaction with two species in one
compartment using initialAssignment to set the compartment size.
componentTags: Compartment, Species, Reaction, InitialAssignment 
testTags:      Amount, NonUnityCompartment, LocalParameters, InitialValueReassigned
testType:      TimeCourse
levels:        2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called "compartment".  There are two
species named S1 and S2.  The model contains one reaction defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $compartment * k * S1$  |]

Reaction S1 -> S2 defines one local parameter k.

The model contains one initialAssignment that assigns the initial value for
compartment:

[{width:30em,margin: 1em auto}| Variable    | Formula      |
 | compartment | $534 * 0.001$  |]

Note: InitialAssignments override any declared initial values.  In this
case the value from the initialAssignment is consistent with the value
attributed to the compartment by the model definition.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1                |$1.5 \x 10^-2$  |mole                      |
|Initial amount of S2                |$          0.0$ |mole                      |
|Value of local parameter k          |$          100$ |second^-1^                |
|Volume of compartment "compartment" |$        0.534$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00027" ];

addCompartment[ compartment, size->0.534];
addSpecies[ S1, initialAmount -> 1.5 10^-2 ];
addSpecies[ S2, initialAmount -> 0.0 ];
addInitialAssignment[ compartment, math -> 534 * 0.001];
addReaction[ S1 -> S2, reversible -> False,
	     kineticLaw -> compartment * k * S1, parameters -> {k -> 100} ];

makemodel[]

