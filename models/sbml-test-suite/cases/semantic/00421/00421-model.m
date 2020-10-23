(* 

category:      Test
synopsis:      Basic two reactions with three species in one 1D compartment
               and two events that assign value to a species, subject to a delay.
componentTags: Compartment, Species, Reaction, Parameter, EventWithDelay 
testTags:      Amount, EventIsPersistent
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are three species
called S1, S2 and S3 and two parameters called k1 and k2.  Compartment C is
1-dimensional.  The model contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 | $k1 * S1 * S2 * C$  |
| S3 -> S1 + S2 | $k2 * S3 * C$     |]

The model contains two events that assign values to species S2 and S1:

[{width:30em,margin: 1em auto}| | *Trigger*    | *Delay* | *Assignments* |
 | Event1                      | $S1 < 0.65$  | $7.5$   | $S2 = 1$      |
 | Event2                      | $S3 > 1.2$   | $4.5$   | $S1 = 1$      |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value* |*Units*                   |
|Initial amount of S1          |$1.0$   |mole                      |
|Initial amount of S2          |$2.0$   |mole                      |
|Initial amount of S3          |$1.0$   |mole                      |
|Value of parameter k1         |$0.075$ |metre mole^-1^ second^-1^ |
|Value of parameter k2         |$0.025$ |second^-1^                |
|Length of compartment C       |$1$     |metre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00421" ];

addCompartment[ C, size -> 1, spatialDimensions -> 1 ];
addSpecies[ S1, initialAmount -> 1.0];
addSpecies[ S2, initialAmount -> 2.0];
addSpecies[ S3, initialAmount -> 1.0];
addParameter[ k1, value -> .075 ];
addParameter[ k2, value -> .025 ];
addReaction[ S1 + S2 -> S3, reversible -> False,
	     kineticLaw -> k1 * S1 * S2 * C ];
addReaction[ S3 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 * C ];
addEvent[ trigger -> S1 < 0.65, delay->7.5, eventAssignment -> S2->1 ];
addEvent[ trigger -> S3 > 1.2, delay->4.5, eventAssignment -> S1->1 ];

makemodel[]

