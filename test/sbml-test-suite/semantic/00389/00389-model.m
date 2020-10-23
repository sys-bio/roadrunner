(* 

category:      Test
synopsis:      Basic two reactions with three species in one compartment
               and one event that assigns value to a species.
componentTags: Compartment, Species, Reaction, EventNoDelay 
testTags:      Amount, LocalParameters
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are three species
called S1, S2 and S3.  The model contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 | $k * S1 * S2 * C$  |
| S3 -> S1 + S2 | $k * S3 * C$     |]

Reaction S1 + S2 -> S3 defines one local parameter k.  Reaction S3 -> S1 +
S2 defines another local parameter k.  Note that these parameters have a
scope local to the defining reaction.

The model contains one event that assigns a value to species S2:

[{width:30em,margin: 1em auto}| | *Trigger*   | *Delay* | *Assignments* |
 | Event1                      | $S1 < 0.75$ | $-$     | $S2 = 1$    |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value* |*Units*  |
|Initial amount of S1          |$1.0$   |mole                      |
|Initial amount of S2          |$2.0$   |mole                      |
|Initial amount of S3          |$1.0$   |mole                      |
|Value of local parameter k    |$0.75$  |litre mole^-1^ second^-1^ |
|Value of local parameter k    |$0.25$  |second^-1^                |
|Volume of compartment C       |$1$     |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00389" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.0];
addSpecies[ S2, initialAmount -> 2.0];
addSpecies[ S3, initialAmount -> 1.0];
addReaction[ S1 + S2 -> S3, reversible -> False,
	     kineticLaw -> k * S1 * S2 * C, parameters -> {k -> 0.75}  ];
addReaction[ S3 -> S1 + S2, reversible -> False,
	     kineticLaw -> k * S3 * C, parameters -> {k -> 0.25}  ];
addEvent[ trigger -> S1 < 0.75, eventAssignment -> S2->1 ];

makemodel[]

