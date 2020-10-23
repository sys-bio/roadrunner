(* 

category:      Test
synopsis:      Basic reaction and rate rule with three species in a compartment. 
componentTags: Compartment, Species, Reaction, Parameter, RateRule 
testTags:      Amount, BoundaryCondition
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

The model contains one compartment called "compartment".  There are three
species named S1, S2 and S3 and two parameters named k1 and k2.  Species S2
is labeled as an SBML boundary species.  The model contains one reaction
defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $k1 * S1 * compartment$  |]

The model contains one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Rate | S3 | $k1 * k2$  |]


The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1                |$1.5 \x 10^-2$ |mole                      |
|Initial amount of S2                |$1.0 \x 10^-2$ |mole                      |
|Initial amount of S3                |$1.0 \x 10^-2$ |mole                      |
|Value of parameter k1               |$         1.75$ |second^-1^ |
|Value of parameter k2               |$1.5 \x 10^-3$ |mole |
|Volume of compartment "compartment" |$            1$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

Note: The test data for this model was generated from an analytical solution
of the system of equations.

*)

newcase[ "00336" ];

addCompartment[ compartment, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.5 10^-2 ];
addSpecies[ S2, initialAmount -> 1.0 10^-2, boundaryCondition->True   ];
addSpecies[ S3, initialAmount -> 1.0 10^-2];
addParameter[ k1, value -> 1.75 ];
addParameter[ k2, value -> 1.5 10^-3 ];
addRule[ type->RateRule, variable -> S3, math -> k1 * k2];
addReaction[ S1 -> S2, reversible -> False,
	     kineticLaw -> k1 * S1 * compartment ];

makemodel[]

