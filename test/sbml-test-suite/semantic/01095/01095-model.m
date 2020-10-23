(* 

category:      Test
synopsis:      Two reactions and a rate rule with four species in a compartment. 
componentTags: InitialAssignment, Compartment, Species, Reaction, Parameter, RateRule, EventWithDelay
testTags:      InitialValueReassigned, Amount, AssignedConstantStoichiometry, NonUnityStoichiometry
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Numeric

Note:  This test is the L3 version of model 731.

The model contains one compartment called C.  There are four
species called S1, S2, S3 and S4 and three parameters called k1, k2 and p1.  The
model contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 | $k1 * S1 * S2 * C$  |
| S3 -> S1 + (4 * p1)S2 | $k2 * S3 * C$  |]

The model contains one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Rate | S4 | $1$  |]

The model contains one event that assigns a value to species S2:

[{width:30em,margin: 1em auto}| | *Trigger*    | *Delay* | *Assignments* |
 | Event1 | $S1 < 0.5$ | $2.5$   | $S2 = 2$    |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1                |$1.5$ |mole                      |
|Initial amount of S2                |$2.0$ |mole                      |
|Initial amount of S3                |$1.5$ |mole                      |
|Initial amount of S4                |$  1$ |mole                      |
|Value of parameter k1               |$0.$ |litre mole^-1^ second^-1^ |
|Value of parameter k2               |$0.00025$ |second^-1^ |
|Value of parameter p1               |$0.25$ |dimensionless |
|Volume of compartment C |$            1$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00731" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.5 ];
addSpecies[ S2, initialAmount -> 2.0];
addSpecies[ S3, initialAmount -> 1.5];
addSpecies[ S4, initialAmount -> 1 ];
addParameter[ k1, value -> .75];
addParameter[ k2, value -> .00025];
addParameter[ p1, value -> 0.25];
addRule[ type->RateRule, variable -> S4, math -> 1];
addReaction[ S1 + S2 -> S3, reversible -> False,
	     kineticLaw -> k1 * S1 * S2 * C ];
addReaction[reactants->{S3}, products->{S1, S2}, productStoichiometry->{1, 4 * p1}, 
       reversible -> False,
	     kineticLaw -> k2 * S3 * C ];
addEvent[ trigger -> S1 < 0.5, delay -> 2.5, eventAssignment -> S2 -> 2.0 ];

makemodel[]

