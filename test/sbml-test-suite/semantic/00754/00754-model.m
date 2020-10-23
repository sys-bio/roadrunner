(* 

category:      Test
synopsis:      Basic two reactions with three species in one compartment
               and one event that assigns value to a species using a function.
componentTags: Compartment, Species, Reaction, Parameter, EventNoDelay, FunctionDefinition, InitialAssignment
testTags:      Amount
testType:      TimeCourse
levels:        2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are three species
called S1, S2 and S3 and five parameters called k1, k2, k3, k4 and p1.  The
model contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 | $k1 * S1 * S2 * C$  |
| S3 -> S1 + S2 | $k2 * S3 * C$     |]

The model contains one event that assigns a value to species S2:

[{width:30em,margin: 1em auto}| | *Trigger*    | *Delay* | *Assignments* |
 | Event1 | $S1 < 0.5$ | $-$   | $S2 = add(k3, k4)$    |]
 
The model contains one initialAssignment:

[{width:30em,margin: 1em auto}| Variable | Formula |
 | S1 | $add(p1, p1)$  |]

Note: SBML's InitialAssignment construct override any declared initial
values.  In this case the initial value declared for species S1 is
consistent with the value returned by the InitialAssignment.

Both the eventAssignment and the initialAssignment use the functionDefinition defined as:
[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | add | x, y      | $x + y$   |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1    |$1.0$  |mole                      |
|Initial amount of S2    |$2.0$  |mole                      |
|Initial amount of S3    |$1.0$  |mole                      |
|Value of parameter k1   |$0.75$           |litre mole^-1^ second^-1^ |
|Value of parameter k2   |$0.25$           |second^-1^                |
|Value of parameter k3   |$0.8$ |mole litre^-1^ |
|Value of parameter k4   |$0.2$ |mole litre^-1^ |
|Value of parameter p1   |$0.5$ |mole |
|Volume of compartment C |$1$              |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00754" ];

addFunction[ add, arguments -> {x, y}, math -> x + y];
addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.0];
addSpecies[ S2, initialAmount -> 2.0];
addSpecies[ S3, initialAmount -> 1.0];
addParameter[ k1, value -> 0.75 ];
addParameter[ k2, value -> 0.25 ];
addParameter[ k3, value -> 0.8 ];
addParameter[ k4, value -> 0.2 ];
addParameter[ p1, value -> 0.5 ];
addInitialAssignment[ S1, math -> add[p1, p1]];
addReaction[ S1 + S2 -> S3, reversible -> False,
	     kineticLaw -> k1 * S1 * S2 * C ];
addReaction[ S3 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 * C ];
addEvent[ trigger -> S1 < 0.5, eventAssignment -> S2->add[k3, k4] ];

makemodel[]

