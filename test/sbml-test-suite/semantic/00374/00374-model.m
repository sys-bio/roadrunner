(* 

category:      Test
synopsis:      Basic two reactions with four species in one compartment
               and an event that assigns value to two species.
componentTags: Compartment, Species, Reaction, Parameter, EventNoDelay 
testTags:      Concentration, NonUnityCompartment
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are four species called
S1, S2, S3 and S4 and two parameters called k1 and k2.  The model contains
two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 + S4 | $k1 * S1 * S2 * C$  |
| S3 + S4 -> S1 + S2 | $k2 * S3 * S4 * C$  |]

The model contains one event that assigns a value to both species S1 and S4:

[{width:30em,margin: 1em auto}| | *Trigger* | *Delay* | *Assignments* |
| Event1                      | $S4 > S2$ | $-$     | $S1 -> 2 \x 10^-4$   |
|                             |           |         | $S4 -> 2 \x 10^-4$ |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value*          |*Units*  |
|Initial amount of S1          |$4.0 \x 10^-4$  |mole                       |
|Initial amount of S2          |$4.8 \x 10^-4$  |mole                       |
|Initial amount of S3          |$8.0 \x 10^-4$  |mole                       |
|Initial amount of S4          |$4.0 \x 10^-4$  |mole                       |
|Value of parameter k1         |$0.75 \x 10^4$  |litre mole^-1^ second^-1^  |
|Value of parameter k2         |$0.025 \x 10^4$  |litre mole^-1^ second^-1^  |
|Volume of compartment C       |$0.25$           |litre                      |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00374" ];

addCompartment[ C, size -> 0.25 ];
addSpecies[ S1, initialConcentration -> 4.0 10^-4];
addSpecies[ S2, initialConcentration -> 4.8 10^-4];
addSpecies[ S3, initialConcentration -> 8.0 10^-4];
addSpecies[ S4, initialConcentration -> 4.0 10^-4];
addParameter[ k1, value -> 0.75 10^4 ];
addParameter[ k2, value -> .025 10^4 ];
addReaction[ S1 + S2 -> S3 + S4, reversible -> False,
	     kineticLaw -> k1 * S1 * S2 * C ];
addReaction[ S3 + S4 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 * S4 * C ];
addEvent[ trigger -> S4 > S2, eventAssignment -> {S1->2/10^4, S4->2/10^4}  ];

makemodel[]

