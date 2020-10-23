(* 

category:      Test
synopsis:      Third order mass action kinetics with two reactants.
componentTags: Compartment, Species, Reaction, Parameter 
testTags:      Amount
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are three species called 
S1, S2 and S3 and one parameter called k.  The model contains one reaction that has
a kinetic law that defines third order mass action kinetics with two reactants
(as referenced by SBO:0000054):

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 | $k * S1 * S1 * S2 * C$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value*       |*Units*  |
|Initial amount of S1        |$1.0$ |mole                      |
|Initial amount of S2        |$0.5$ |mole                      |
|Initial amount of S2        |$0$ |mole                      |
|Value of parameter k       |$1.1$          |litre^2^ mole^-2^ second^-1^ |
|Volume of compartment C     |$1$             |litre                  |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00807" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.0];
addSpecies[ S2, initialAmount -> 0.5];
addSpecies[ S3, initialAmount -> 0];
addParameter[ k, value -> 1.1 ];
addReaction[ S1 + S2 -> S3, reversible -> False,
	     kineticLaw -> k * S1 * S1 * S2 * C ];

makemodel[]

