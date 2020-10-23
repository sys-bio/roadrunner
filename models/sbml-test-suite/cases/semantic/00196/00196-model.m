(* 

category:      Test
synopsis:      Basic single forward reaction with two species in one
compartment.
componentTags: Compartment, Species, Reaction, Parameter 
testTags:      Amount
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called "compartment".  There are two
species named S1 and S2 and two parameters named p1 and p2.  The model
contains one reaction defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $Floor(S1 * p1) / p2$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1                |$          1.0$ |mole                      |
|Initial amount of S2                |$          0.0$ |mole                      |
|Value of parameter p1               |$         10.5$ |litre |
|Value of parameter p2               |$           10$ |second |
|Volume of compartment "compartment" |$           1.$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00196" ];

addCompartment[ compartment, size -> 1.0];
addSpecies[ S1, initialAmount -> 1.0];
addSpecies[ S2, initialAmount -> 0.0];
addParameter[ p1, value -> 10.5 ];
addParameter[ p2, value -> 10 ];
addReaction[ S1 -> S2, reversible -> False,
	     kineticLaw -> Floor[S1 * p1]/p2 ];

makemodel[]

