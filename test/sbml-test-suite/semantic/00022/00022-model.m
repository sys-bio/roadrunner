(* 

category:      Test
synopsis:      Two reactions with two species in one compartment, with a
               reaction involving a non-integer stoichiometry.
componentTags: Compartment, Species, Reaction, Parameter 
testTags:      Amount, NonUnityCompartment, NonUnityStoichiometry
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

The model contains one compartment called "compartment".  There are two
species named S1 and S2 and two parameters named k1 and k2.  The model
contains two reactions defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> 0.3 S2 | $k1 * S1 * compartment$  |
| 0.7 S2 -> S1 | $k2 * S2 * compartment$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1                |$1.5 \x 10^-3$  |mole           |
|Initial amount of S2                |$0 \x$           |mole           |
|Value of parameter k1               |$0.35$           |second^-1^     |
|Value of parameter k2               |$180$            |second^-1^     |
|Volume of compartment "compartment" |$0.3$            |litre          |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

newcase[ "00022" ];

addCompartment[ compartment, size -> 0.3 ];
addSpecies[ S1, initialAmount -> 1.5 10^-3 ];
addSpecies[ S2, initialAmount -> 0 ];
addParameter[ k1, value -> 0.35 ];
addParameter[ k2, value -> 180 ];
addReaction[ S1 -> 0.3 S2, reversible -> False,
	     kineticLaw -> k1 * S1 * compartment ];
addReaction[ 0.7 S2 -> S1, reversible -> False,
	     kineticLaw -> k2 * S2 * compartment ];

makemodel[]

