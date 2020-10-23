(* 

category:      Test
synopsis:      Single reversible reaction with an initial assignment.
componentTags: Compartment, Species, Reaction, Parameter, InitialAssignment  
testTags:      Amount, InitialValueReassigned
testType:      TimeCourse
levels:        2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are three species called 
S1, S2 and S3 and three parameters called kf, kr and p1.  The model contains  one reaction
defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 <-> S3 | $(kf * S1 * S2 - kr * S3) * C$  |]

The model contains one initialAssignment:

[{width:30em,margin: 1em auto}| Variable | Formula |
 | S1 | $p1 * 2$  |]

Note: SBML's InitialAssignment construct override any declared initial
values.  In this case the initial value declared for species S1 is
inconsistent with the value returned by the InitialAssignment.  The
calculated value should be used.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value*       |*Units*  |
|Initial amount of S1        |$0$ |mole                      |
|Initial amount of S2        |$0.5$ |mole                      |
|Initial amount of S3        |$0$ |mole                      |
|Value of parameter kf       |$1.1$          |litre mole^-1^ second^-1^ |
|Value of parameter kr       |$0.09$          |second^-1^ |
|Value of parameter p1       |$0.5$           |mole                |
|Volume of compartment C     |$1$             |litre                  |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00836" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialAmount -> 0.0];
addSpecies[ S2, initialAmount -> 0.5];
addSpecies[ S3, initialAmount -> 0.0];
addParameter[ kf, value -> 1.1 ];
addParameter[ kr, value -> 0.09 ];
addParameter[ p1, value -> 0.5 ];
addInitialAssignment[ S1, math -> p1*2];
addReaction[ S1 + S2 -> S3,
	     kineticLaw -> (kf * S1 * S2 - kr * S3) * C ];

makemodel[]

