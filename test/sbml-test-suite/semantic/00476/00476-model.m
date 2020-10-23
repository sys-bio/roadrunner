(* 

category:      Test
synopsis:      Basic two reactions with three species in a compartment using 
initialAssignment to set the initial value of one species.
componentTags: Compartment, Species, Reaction, Parameter, InitialAssignment 
testTags:      Amount, InitialValueReassigned
testType:      TimeCourse
levels:        2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are three species called 
S1, S2 and S3 and three parameters called k1, k2 and p1.  The model contains 
two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 | $k1 * S1 * S2 * C$  |
| S3 -> S1 + S2 | $k2 * S3 * C$  |]

  The model contains one initialAssignment:

[{width:30em,margin: 1em auto}| Variable | Formula |
 | S1 | $p1 * 2$  |]

Note: SBML's InitialAssignment construct override any declared initial
values.  In this model, the initial value for the species S1 has not been
explicitly declared and must be calculated using the InitialAssignment.

The initial conditions are as follows:

[{width 30em,margin: 1em auto}| |*Value*        |*Units*  |
|Initial amount of S1        |$undeclared$     |mole                      |
|Initial amount of S2        |$2.0 \x 10^-1$  |mole                      |
|Initial amount of S3        |$1.0 \x 10^-1$  |mole                      |
|Value of parameter k1       |$0.75$           |litre mole^-1^ second^-1^ |
|Value of parameter k2       |$0.25$           |second^-1^                |
|Value of parameter p1       |$1.25 \x 10^-2$ |mole                |
|Volume of compartment C     |$1$              |litre                  |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) they must be treated as concentrations where they appear in
expressions.

*)

newcase[ "00476" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1];
addSpecies[ S2, initialAmount -> 2.0 10^-1];
addSpecies[ S3, initialAmount -> 1.0 10^-1];
addParameter[ k1, value -> 0.75 ];
addParameter[ k2, value -> 0.25 ];
addParameter[ p1, value -> 0.125 10^-1 ];
addInitialAssignment[ S1, math -> p1*2];
addReaction[ S1 + S2 -> S3, reversible -> False,
	     kineticLaw -> k1 * S1 * S2 * C ];
addReaction[ S3 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 * C ];

makemodel[]

