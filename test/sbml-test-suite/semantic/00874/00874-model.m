(* 

category:      Test
synopsis:      Two reactions in one compartment 
with one reaction marked as fast and one species as boundary.
componentTags: Compartment, Species, Reaction, Parameter 
testTags:      Amount, FastReaction, BoundaryCondition
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1
generatedBy:   Numeric

The model contains one compartment called C.  There are four
species called A1, A2, A3 and A4 and four parameters called Kf1, Kr1, Kf
and Kr.  Species A3 is labeled as an SBML boundary species.
The model contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| A4 -> A2 | $Kf1 * A4 * C$  |
| A1 + A2 -> A3 | $Kf * A1 * A2 * C$  |] 
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


newcase[ "00874" ];

addCompartment[ C, size -> 1 ];
addSpecies[ A1, initialAmount -> 2 ];
addSpecies[ A2, initialAmount -> 3 ];
addSpecies[ A3, initialAmount -> 4, boundaryCondition -> True ];
addSpecies[ A4, initialAmount -> 1 ];
addParameter[ Kf1, value -> 1 ];
addParameter[ Kf, value -> 1 ];
addReaction[ A4 -> A2, fast -> True, reversible -> False,
 kineticLaw -> Kf1 * A4 * C ];
addReaction[ A1 + A2 -> A3, reversible ->False,
 kineticLaw -> Kf * A1 * A2 * C ];

makemodel[]
