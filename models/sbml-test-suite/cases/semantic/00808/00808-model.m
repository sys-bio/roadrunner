(* 

category:      Test
synopsis:      Third order mass action kinetics with three reactants.
componentTags: Compartment, Species, Reaction, Parameter 
testTags:      Amount
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called C.  There are four species called 
S1, S2, S3 and S4 and one parameter called k.  The model contains one reaction that has
a kinetic law that defines third order mass action kinetics with three reactants
(as referenced by SBO:0000061):

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 + S3 -> S4 | $k * S1 * S2 * S3 * C$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value*       |*Units*  |
|Initial amount of S1        |$1.0$ |mole                      |
|Initial amount of S2        |$0.5$ |mole                      |
|Initial amount of S3        |$0.75$ |mole                      |
|Initial amount of S4        |$0$ |mole                      |
|Value of parameter k       |$0.8$          |litre^2^ mole^-2^ second^-1^ |
|Volume of compartment C     |$1$             |litre                  |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00808" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.0];
addSpecies[ S2, initialAmount -> 0.5];
addSpecies[ S3, initialAmount -> 0.75];
addSpecies[ S4, initialAmount -> 0];
addParameter[ k, value -> 0.8 ];
addReaction[ S1 + S2 + S3 -> S4, reversible -> False,
	     kineticLaw -> k * S1 * S2 * S3 * C ];

makemodel[]

