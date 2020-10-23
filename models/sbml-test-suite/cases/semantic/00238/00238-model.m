(* 

category:      Test
synopsis:      Basic single forward reaction with two species in a 0-dimensional
compartment, with a species acting as a boundary condition.
componentTags: Compartment, Species, Reaction, Parameter 
testTags:      Amount, 0D-Compartment, BoundaryCondition
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

The model contains one compartment called "compartment".  There are two
species named S1 and S2 and one parameter named k1.  Compartment
"compartment" is 0-dimensional.  Species S1 is labeled as an SBML boundary
species.  The model contains one reaction defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $k1 * S1$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|      |*Value*          |*Units*  |
|Initial amount of S1  |$1.5 \x 10^-1$ |mole                      |
|Initial amount of S2  |$            0$ |mole                      |
|Value of parameter k1 |$        1.023$ |second^-1^ |]

In this example the compartment has its spatialDimensions attribute set to
zero; i.e., it represents a point and therefore cannot have size or units.
The species values must be treated as amounts and not concentrations.

Note: The test data for this model was generated from an analytical solution
of the system of equations.

*)

newcase[ "00238" ];

addCompartment[ compartment, spatialDimensions-> 0 ];
addSpecies[ S1, initialAmount -> 1.5 10^-1, boundaryCondition -> True ];
addSpecies[ S2, initialAmount -> 0 ];
addParameter[ k1, value -> 1.023 ];
addReaction[ S1 -> S2, reversible -> False,
	     kineticLaw -> k1 * S1 ];

makemodel[]

