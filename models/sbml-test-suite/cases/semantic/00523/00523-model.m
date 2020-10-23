(* 

category:      Test
synopsis:      Basic two reactions with three species in a compartment using 
initialAssignment to set the initial value of one species.
componentTags: Compartment, Species, Reaction, Parameter, InitialAssignment 
testTags:      Amount, LocalParameters
testType:      TimeCourse
levels:        2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are three species called 
S1, S2 and S3 and one parameters called p1.  The model contains 
two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 | $k * S1 * S2 * C$  |
| S3 -> S1 + S2 | $k * S3 * C$  |]

Reaction $S1 + S2 -> S3$ defines one local parameter k.  Reaction $S3 -> S1
+ S2$ defines another (different) local parameter k.  Note that these
parameters have a scope local to the defining reaction.

The model contains one initialAssignment:

[{width:30em,margin: 1em auto}| Variable | Formula |
 | S1 | $p1 * 2$  |]

Note: SBML's InitialAssignment construct override any declared initial
values.  In this case the initial value declared for species S1 is
consistent with the value returned by the InitialAssignment.

The initial conditions are as follows:

[{width 30em,margin: 1em auto}| |*Value*        |*Units*  |
|Initial amount of S1        |$0.25 \x 10^-2$  |mole                      |
|Initial amount of S2        |$2.0 \x 10^-2$  |mole                      |
|Initial amount of S3        |$1.0 \x 10^-2$  |mole                      |
|Value of parameter p1       |$1.25 \x 10^-3$ |mole                |
|Value of parameter local k  |$0.75$           |litre mole^-1^ second^-1^ |
|Value of parameter local k  |$0.25$           |second^-1^                |
|Volume of compartment C     |$1$              |litre                  |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) they must be treated as concentrations where they appear in
expressions.

*)

newcase[ "00523" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialAmount -> 0.25 10^-2];
addSpecies[ S2, initialAmount -> 2.0 10^-2];
addSpecies[ S3, initialAmount -> 1.0 10^-2];
addParameter[ p1, value -> 0.125 10^-2 ];
addInitialAssignment[ S1, math -> p1*2];
addReaction[ S1 + S2 -> S3, reversible -> False,
	     kineticLaw -> k * S1 * S2 * C, parameters -> {k -> 0.75}  ];
addReaction[ S3 -> S1 + S2, reversible -> False,
	     kineticLaw -> k * S3 * C, parameters -> {k -> 0.25}   ];

makemodel[]

