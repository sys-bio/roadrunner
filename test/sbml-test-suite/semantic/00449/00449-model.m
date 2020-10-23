(* 

category:      Test
synopsis:      Basic two reactions with four species in one compartment
               and two events that assign value to a species, subject to a delay.
componentTags: Compartment, Species, Reaction, EventWithDelay 
testTags:      Amount, LocalParameters
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are four species called
S1, S2, S3 and S4.  The model contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 + S4 | $k * S1 * S2 * C$  |
| S3 + S4 -> S1 + S2 | $k * S3 * S4 * C$  |]

Reaction $S1 + S2 -> S3 + S4$ defines one local parameter k.  Reaction $S3
+ S4 -> S1 + S2$ defines another (different) local parameter k.  Note that
these parameters have a scope local to the defining reaction.

The model contains two events that assign value to species S1 and S4:

[{width:30em,margin: 1em auto}| | *Trigger*             | *Delay* | *Assignments*       |
 | Event1                      | $S4 > S2$             | $1.5$   | $S1 -> 2 \x 10^-4$ |]
 | Event2                      | $S3 > 2.25 \x 10^-4$ | $0.5$   | $S4 -> 1 \x 10^-4$ |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value*          |*Units*  |
|Initial amount of S1          |$1.0 \x 10^-4$  |mole                       |
|Initial amount of S2          |$1.0 \x 10^-4$  |mole                       |
|Initial amount of S3          |$2.0 \x 10^-4$  |mole                       |
|Initial amount of S4          |$1.0 \x 10^-4$  |mole                       |
|Value of local parameter k    |$0.75 \x 10^4$  |litre mole^-1^ second^-1^  |
|Value of local parameter k    |$0.25 \x 10^4$  |litre mole^-1^ second^-1^  |
|Volume of compartment C       |$1$              |litre                      |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00449" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.0 10^-4];
addSpecies[ S2, initialAmount -> 1.2 10^-4];
addSpecies[ S3, initialAmount -> 2.0 10^-4];
addSpecies[ S4, initialAmount -> 1.0 10^-4];
addReaction[ S1 + S2 -> S3 + S4, reversible -> False,
	     kineticLaw -> k * S1 * S2 * C, parameters -> {k -> 7500}  ];
addReaction[ S3 + S4 -> S1 + S2, reversible -> False,
	     kineticLaw -> k * S3 * S4 * C , parameters -> {k -> 2500} ];
addEvent[ trigger -> S4 > S2, delay->1.5, eventAssignment -> S1->2 10^-4 ];
addEvent[ trigger -> S3 > 9/(4 10^4), delay->0.5, eventAssignment -> S4->1 10^-4 ];

makemodel[]

