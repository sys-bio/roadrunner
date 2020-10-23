(* 

category:      Test
synopsis:      Two reversible reactions with four species in one compartment
               and one event that assigns value to two species subject to a delay.
componentTags: Compartment, Species, Reaction, Parameter, EventWithDelay 
testTags:      Amount, ReversibleReaction, EventUsesTriggerTimeValues
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are four species
called S1, S2, S3 and S4 and four parameters called k1, k2, kf and kr.  The model
contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 <-> S2 | $(kf * S1 - kr * S2) * C$  |
| S3 <-> S4 | $(k1 * S3 - k2 * S4) * C$  |]

The model contains one event that assigns values to both species S2 and S4:

[{width:30em,margin: 1em auto}| | *Trigger*   | *Delay* | *Assignments* |
 | Event1                      | $S1 < 0.5$ | $1.89$     | $S2 = 1.5$    |
 |                             |             |         | $S4 = S2$     |]
 
Note that the event assignments should happen simultaneously, not
sequentially; i.e., the value of S2 assigned to S4 is the value at the
point at which the event was triggered.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value* |*Units*  |
|Initial amount of S1          |$1.0$   |mole                      |
|Initial amount of S2          |$2.0$   |mole                      |
|Initial amount of S3          |$1.0$   |mole                      |
|Initial amount of S4          |$1.5$   |mole                      |
|Value of parameter k1         |$0.75$  |second^-1^                |
|Value of parameter k2         |$0.15$  |second^-1^                |
|Value of parameter kf         |$0.9$  |second^-1^                |
|Value of parameter kr         |$0.075$  |second^-1^                |
|Volume of compartment C       |$1$     |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00849" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.0];
addSpecies[ S2, initialAmount -> 2.0];
addSpecies[ S3, initialAmount -> 1.0];
addSpecies[ S4, initialAmount -> 1.5];
addParameter[ k1, value -> 0.75 ];
addParameter[ k2, value -> 0.15 ];
addParameter[ kf, value -> 0.9 ];
addParameter[ kr, value -> 0.075 ];
addReaction[ S1 -> S2,
	     kineticLaw -> (kf * S1 - kr * S2) * C ];
addReaction[ S3 -> S4,
	     kineticLaw -> (k1 * S3 - k2 * S4) * C ];
addEvent[ trigger -> S1 < 0.5, delay -> 1.89, eventAssignment -> {S2->1.5, S4->S2} ];

makemodel[]

