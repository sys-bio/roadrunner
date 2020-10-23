(* 

category:      Test
synopsis:      A simple reaction during which the species values are reset 
               by events.
componentTags: Compartment, Species, Reaction, Parameter, EventNoDelay 
testTags:      Amount
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called "compartment".  There are two
species named S1 and S2 and one parameter named k1.  The model contains one
reaction defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $k1 * S1 * compartment$  |]

The model contains two events defined as:

[{width:30em,margin: 1em auto}| | *Trigger*    | *Delay* | *Assignments* |
 | Event1 | $S1 < 0.1$ | $-$   | $S1 = 1$ |
 | Event2 | $S2 > 0.5$ | $-$   | $S2 = 0$ |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*         |*Units*  |
|Initial amount of S1                |$            1$ |mole                      |
|Initial amount of S2                |$            0$ |mole                      |
|Value of parameter k1               |$            1$ |second^-1^                |
|Volume of compartment "compartment" |$            1$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00041" ];

addCompartment[ compartment, size -> 1 ];
addSpecies[ S1, initialAmount -> 1];
addSpecies[ S2, initialAmount -> 0];
addParameter[ k1, value -> 1 ];
addReaction[ S1 -> S2, reversible -> False,
	     kineticLaw -> k1 * S1 * compartment ];
addEvent[ trigger -> S1 < 0.1, eventAssignment -> S1->1];
addEvent[ trigger -> S2 > 0.5, eventAssignment -> S2->0];

makemodel[]

