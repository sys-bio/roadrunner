(* 

category:      Test
synopsis:      Basic two reactions with three species in one compartment
and two events that assigns value to a species with a delay using csymbol time.
componentTags: Compartment, Species, Reaction, Parameter, CSymbolTime, EventWithDelay, EventNoDelay 
testTags:      Amount, EventIsPersistent
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are three species
called S1, S2 and S3 and three parameters called k1, k2 and k3.  The model
contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 | $k1 * S1 * S2 * C$  |
| S3 -> S1 + S2 | $k2 * S3 * C$     |]

The model contains two events that assign values to species S2 and S1:

[{width:30em,margin: 1em auto}| | *Trigger*    | *Delay* | *Assignments* |
 | Event1 | $S1 < 0.75$ | $k3 * time$   | $S2 = 1$    |
 | Event2 | $S3 > 1.4$ | $-$   | $S1 = 0.25$    |]
where the symbol 'time' denotes the current simulation time.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1    |$1.0$  |mole                      |
|Initial amount of S2    |$2.0$  |mole                      |
|Initial amount of S3    |$1.0$  |mole                      |
|Value of parameter k1   |$0.75$           |litre mole^-1^ second^-1^ |
|Value of parameter k2   |$0.25$           |second^-1^                |
|Value of parameter k3   |$4$           |dimensionless                |
|Volume of compartment C |$1$              |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00887" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.0];
addSpecies[ S2, initialAmount -> 2.0];
addSpecies[ S3, initialAmount -> 1.0];
addParameter[ k1, value -> 0.75 ];
addParameter[ k2, value -> 0.25 ];
addParameter[ k3, value -> 4 ];
addReaction[ S1 + S2 -> S3, reversible -> False,
	     kineticLaw -> k1 * S1 * S2 * C ];
addReaction[ S3 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 * C ];
addEvent[ trigger -> S1 < 0.75, 
          delay ->  k3 * \[LeftAngleBracket]time, "time"\[RightAngleBracket],
          eventAssignment -> S2->1  ];
addEvent[ trigger -> S3 > 1.4, eventAssignment -> S1->0.25  ];

makemodel[]

