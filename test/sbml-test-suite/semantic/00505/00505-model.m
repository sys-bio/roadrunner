(* 

category:      Test
synopsis:      Basic two reactions with four species in a compartment using 
initialAssignment to set the initial value of one species.
componentTags: Compartment, Species, Reaction, Parameter, InitialAssignment 
testTags:      Amount, ConstantSpecies
testType:      TimeCourse
levels:        2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are four species called 
S1, S2, S3 and S4 and three parameters called k1, k2 and p1.  Species S4 is labeled 
as a constant species.   The model contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 | $k1 * S1 * S4 * C$  |
| S3 -> S1 + S2 | $k2 * S3 * C$  |]

  The model contains one initialAssignment:

[{width:30em,margin: 1em auto}| Variable | Formula |
 | S4 | $p1 * 2$  |]

Note: SBML's InitialAssignment construct override any declared initial
values.  In this case the initial value declared for species S4 is
consistent with the value returned by the InitialAssignment.

The initial conditions are as follows:

[{width 30em,margin: 1em auto}| |*Value*        |*Units*  |
|Initial amount of S1        |$0.25 \x 10^-3$ |mole                      |
|Initial amount of S2        |$2.0 \x 10^-3$  |mole                      |
|Initial amount of S3        |$1.0 \x 10^-3$  |mole                      |
|Initial amount of S4        |$1.0 \x 10^-3$     |mole                      |
|Value of parameter k1       |$0.75$           |litre mole^-1^ second^-1^ |
|Value of parameter k2       |$0.25$           |second^-1^                |
|Value of parameter p1       |$5 \x 10^-4$    |mole                |
|Volume of compartment C     |$1$              |litre                  |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) they must be treated as concentrations where they appear in
expressions.

*)

newcase[ "00505" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialAmount -> 0.25 10^-3];
addSpecies[ S2, initialAmount -> 2.0 10^-3];
addSpecies[ S3, initialAmount -> 1.0 10^-3];
addSpecies[ S4, initialAmount -> 1.0 10^-3, constant -> True]
addParameter[ k1, value -> 0.75 ];
addParameter[ k2, value -> 0.25 ];
addParameter[ p1, value -> 0.5 10^-3 ];
addInitialAssignment[ S4, math -> p1*2];
addReaction[ S1 + S2 -> S3, reversible -> False,
	     kineticLaw -> k1 * S1 * S4 * C ];
addReaction[ S3 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 * C ];

makemodel[]

