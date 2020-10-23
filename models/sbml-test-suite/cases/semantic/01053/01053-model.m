(* 

category:      Test
synopsis:      Two reversible reactions in one compartment 
with one reaction marked as fast and one species a boundary.
componentTags: Compartment, Species, Reaction, Parameter 
testTags:      Amount, FastReaction, BoundaryCondition
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1
generatedBy:   Numeric

Note:  This test is a copy of model 873, with the products and reactants reversed, and a negative kinetic law.  This gives the same results, but during the simulation, the kinetic law fails to go negative (a requirement for the ReversibeReaction tag).  The reaction listed below is therfore equivalent to the one in the file, but reversed from it.

The model contains one compartment called C.  There are four
species called A1, A2, A3 and A4 and four parameters called Kf1, Kr1, Kf
and Kr.  Species A2 is labeled as an SBML boundary species.
The model contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| A4 <-> A2 | $(Kf1 * A4 - Kr1 * A2) * C$  |
| A1 + A2 <-> A3 | $(Kf * A1 * A2 - Kr * A3) * C$  |] 
where the first reaction is marked as 'fast' and therefore should be assumed 
to be operating on a time scale significantly faster than the second reaction.
Fast reactions are considered to be instantaneous relative to slow
reactions.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of A1                |$2$  |mole           |
|Initial amount of A2                |$3$              |mole           |
|Initial amount of A3                |$4$              |mole           |
|Initial amount of A4                |$1$              |mole           |
|Value of parameter Kf1               |$1$            |second^-1^     |
|Value of parameter Kr1               |$2$            |second^-1^     |
|Value of parameter Kf               |$1$            |litre mole^-1^ second^-1^     |
|Value of parameter Kr               |$2$              |second^-1^     |
|Volume of compartment C |$1$              |litre          |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)


newcase[ "00873" ];

addCompartment[ C, size -> 1 ];
addSpecies[ A1, initialAmount -> 2];
addSpecies[ A2, initialAmount -> 3, boundaryCondition -> True  ];
addSpecies[ A3, initialAmount -> 4 ];
addSpecies[ A4, initialAmount -> 1 ];
addParameter[ Kf1, value -> 1 ];
addParameter[ Kr1, value -> 2 ];
addParameter[ Kf, value -> 1 ];
addParameter[ Kr, value -> 2 ];
addReaction[ A4 -> A2, fast -> True, 
 kineticLaw -> (Kf1 * A4 - Kr1 * A2) * C ];
addReaction[ A1 + A2 -> A3, 
 kineticLaw -> (Kf * A1 * A2 - Kr * A3) * C ];

makemodel[]
