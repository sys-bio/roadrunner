(* 

category:      Test
synopsis:      Basic two reactions with four species in a 2-dimensional
compartment, with two species acting as a boundary condition.
componentTags: Compartment, Species, Reaction, Parameter 
testTags:      Amount, BoundaryCondition
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called "compartment".  There are four
species named S1, S2, S3 and S4 and two parameters named k1 and k2.
Compartment "compartment" is 2-dimensional.  Both species S1 and S2 are
labeled as an SBML boundary species.  The model contains two reactions
defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 + S2 -> S3 + S4 | $k1 * S1 * S2 * compartment$  |
| S3 + S4 -> S1 + S2 | $k2 * S3 * S4 * compartment$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1              |$1.0 \x 10^-4$ |mole                      |
|Initial amount of S2              |$1.0 \x 10^-4$ |mole                      |
|Initial amount of S3              |$2.0 \x 10^-4$ |mole                      |
|Initial amount of S4              |$1.0 \x 10^-4$ |mole                      |
|Value of parameter k1             |$0.75 \x 10^4$ |metre^2^ mole^-1^ second^-1^ |
|Value of parameter k2             |$0.25 \x 10^4$ |metre^2^ mole^-1^ second^-1^ |
|Area of compartment "compartment" |$            1$ |metre^2^                  |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00220" ];

addCompartment[ compartment, spatialDimensions-> 2, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.0 10^-4, boundaryCondition->True];
addSpecies[ S2, initialAmount -> 1.0 10^-4, boundaryCondition->True];
addSpecies[ S3, initialAmount -> 2.0 10^-4];
addSpecies[ S4, initialAmount -> 1.0 10^-4];
addParameter[ k1, value -> 0.75 10^4 ];
addParameter[ k2, value -> 0.25 10^4 ];
addReaction[ S1 + S2 -> S3 + S4, reversible -> False,
	     kineticLaw -> k1 * S1 * S2 * compartment ];
addReaction[ S3 + S4 -> S1 + S2, reversible -> False,
	     kineticLaw -> k2 * S3 * S4 * compartment ];

makemodel[]

