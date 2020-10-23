(* 

category:      Test
synopsis:      Basic single forward reaction with two species in one
compartment using initialAssignment to set the initial value of the compartment.
componentTags: Compartment, Species, Reaction, Parameter, InitialAssignment 
testTags:      Amount, LocalParameters, InitialValueReassigned
testType:      TimeCourse
levels:        2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are two species called 
S1 and S2 and one parameter called k.  The model contains one 
reaction defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $C * k * S1$  |]

Reaction $S1 -> S2$ defines one local parameter k.  Note that this
parameter has a scope local to the defining reaction.

The model contains one initialAssignment:

[{width:30em,margin: 1em auto}| Variable | Formula |
 | C | $k / 50$  |]

Note: SBML's InitialAssignment construct override any declared initial
values.  In this case the initial value declared for compartment C is
inconsistent with the value returned by the InitialAssignment.


The initial conditions are as follows:

[{width 30em,margin: 1em auto}| |*Value*  |*Units*  |
|Initial amount of S1        |$1$        |mole                      |
|Initial amount of S2        |$1.5$      |mole                      |
|Value of parameter k        |$50$       |litre |
|Value of local parameter k  |$0.5$      |second^-1^ |
|Volume of compartment C     |$5.4$      |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) they must be treated as concentrations where they appear in
expressions.

*)

newcase[ "00525" ];

addCompartment[ C, size -> 5.4, constant -> False];
addSpecies[ S1, initialAmount -> 1 ];
addSpecies[ S2, initialAmount -> 1.5 ];
addParameter[ k, value -> 50 ];
addInitialAssignment[ C, math -> k/50];
addReaction[ S1 -> S2, reversible -> False,
	     kineticLaw -> C * k * S1, parameters -> {k -> 0.5} ];

makemodel[]

