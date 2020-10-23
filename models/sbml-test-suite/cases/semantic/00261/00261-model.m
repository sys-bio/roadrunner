(* 

category:      Test
synopsis:      Basic two reactions involving two species in one
               1 dimensional compartment, with non-unity stoichiometries.
componentTags: Compartment, Species, Reaction, Parameter 
testTags:      Amount, NonUnityStoichiometry
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called "compartment".  There are two
species named S1 and S2 and two parameters named k1 and k2.  Compartment
"compartment" is 1-dimensional.  The model contains two reactions defined
as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> 2 S2 | $k1 * S1 * compartment$    |
| 2 S2 -> S1 | $k2 * S2 * S2 * compartment$ |]

Note the stoichiometry of S2 is 2 in both reactions.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1                |$1.5 \x 10^-2$ |mole                      |
|Initial amount of S2                |$            0$ |mole                      |
|Value of parameter k1               |$         0.35$ |second^-1^ |
|Value of parameter k2               |$          180$ |metre mole^-1^ second^-1^ |
|Length of compartment "compartment" |$            1$ |metre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00261" ];

addCompartment[ compartment, spatialDimensions -> 1, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.5 10^-2 ];
addSpecies[ S2, initialAmount -> 0 ];
addParameter[ k1, value -> 0.35 ];
addParameter[ k2, value -> 180 ];
addReaction[ S1 -> 2 S2, reversible -> False,
	     kineticLaw -> k1 * S1 * compartment ];
addReaction[ 2 S2 -> S1, reversible -> False,
	     kineticLaw -> k2 * S2 * S2 * compartment ];

makemodel[]

