(* 

category:      Test
synopsis:      Single reversible reaction.
componentTags: Compartment, Species, Reaction, Parameter 
testTags:      Amount, ReversibleReaction
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

Note:  This test is a copy of model 811, with the products and reactants reversed, and a negative kinetic law.  This gives the same results, but during the simulation, the kinetic law actually goes negative (a requirement for reversibe reactions).  The reaction listed below is therfore equivalent to the one in the file, but reversed from it.

The model contains one compartment called C.  There are three species called 
S1, S2 and S3 and two parameters called kf and kr.  The model contains  one reaction
defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 <-> S2 + S3 | $(kf * S1 - kr * S2 * S3) * C$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value*       |*Units*  |
|Initial amount of S1        |$1.0$ |mole                      |
|Initial amount of S2        |$0.5$ |mole                      |
|Initial amount of S3        |$0$ |mole                      |
|Value of parameter kf       |$2.5$          |second^-1^ |
|Value of parameter kr       |$0.2$          |litre mole^-1^ second^-1^ |
|Volume of compartment C     |$1$             |litre                  |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00811" ];

addCompartment[ C, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.0];
addSpecies[ S2, initialAmount -> 0.5];
addSpecies[ S3, initialAmount -> 0.0];
addParameter[ kf, value -> 2.5 ];
addParameter[ kr, value -> 0.2 ];
addReaction[ S1 -> S2 + S3,
	     kineticLaw -> (kf * S1 - kr * S2 * S3) * C ];

makemodel[]

