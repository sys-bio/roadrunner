(* 

category:      Test
synopsis:      Basic two reactions with four species in one compartment
               and two events that assign value to a species.
componentTags: InitialAssignment, Compartment, Species, Reaction, Parameter, EventNoDelay 
testTags:      InitialValueReassigned, Amount, AssignedConstantStoichiometry, NonUnityStoichiometry
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Numeric

Note:  This test is the L3 version of model 391.

The model contains one compartment called C.  There are four species called
S1, S2, S3 and S4 and two parameters called k1 and k2.  The model contains
two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> (4 * p1)S3 + S4 | $k1 * S1 * S2 * C$  |
| 2S3 + S4 -> S1 + S2 | $k2 * S3 * S4 * C$  |]

The model contains two events that assign values to species S1 and S4:

[{width:30em,margin: 1em auto}| | *Trigger*            | *Delay* | *Assignments* |
 | Event1                      | $S4 > S2$            | $-$     | $S1 -> 2 \x 10^-4$    |]
 | Event2                      | $S3 > 2.5 \x 10^-4$ | $-$     | $S4 -> 1 \x 10^-4$    |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value*          |*Units*  |
|Initial amount of S1          |$1.0 \x 10^-4$  |mole                       |
|Initial amount of S2          |$1.0 \x 10^-4$  |mole                       |
|Initial amount of S3          |$2.0 \x 10^-4$  |mole                       |
|Initial amount of S4          |$1.0 \x 10^-4$  |mole                       |
|Value of parameter k1         |$0.75 \x 10^4$  |litre mole^-1^ second^-1^  |
|Value of parameter k2         |$0.25 \x 10^4$  |litre mole^-1^ second^-1^  |
|Value of parameter p1         |$          0.5$  |dimensionless              |
|Volume of compartment C       |$1$              |litre                      |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00391" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.0 10^-4];
addSpecies[ S2, initialAmount -> 1.2 10^-4];
addSpecies[ S3, initialAmount -> 2.0 10^-4];
addSpecies[ S4, initialAmount -> 1.0 10^-4];
addParameter[ k1, value -> 0.75 10^4 ];
addParameter[ k2, value -> 0.25 10^4 ];
addParameter[ p1, value -> 0.5 ];
addReaction[ reactants->{S1, S2}, products->{S3, S4}, productStoichiometry->{4 * p1, 1},
       reversible -> False,
	     kineticLaw -> k1 * S1 * S2 * C ];
addReaction[ 2S3 + S4 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 * S4 * C ];
addEvent[ trigger -> S4 > S2, eventAssignment -> S1->2/10^4 ];
addEvent[ trigger -> S3 > 10/(4 10^4), eventAssignment -> S4->1/10^4 ];

makemodel[]

