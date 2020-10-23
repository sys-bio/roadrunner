(* 

category:      Test
synopsis:      One reactions and two rate rules with four species in a compartment. 
componentTags: Compartment, Species, Reaction, Parameter, RateRule 
testTags:      Amount, BoundaryCondition
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

The model contains one compartment called "compartment".  There are four
species named S1, S2, S3 and S4 and two parameters named k1 and k2.
Species S3 is labeled as an SBML boundary species.  The model contains one
reaction defined as:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S1 -> S2 | $k1 * S1 * compartment$  |]

The model contains two rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Rate | S3 | $k1  *  0.5 10^-3$  |
 | Rate | S4 | $-k2  *  0.5 10^-3$  |]


The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial amount of S1                |$ 1.5 \x 10^-3$ |mole                      |
|Initial amount of S2                |$ 2.0 \x 10^-3$ |mole                      |
|Initial amount of S3                |$ 1.5 \x 10^-3$ |mole                      |
|Initial amount of S4                |$   4 \x 10^-3$ |mole                      |
|Value of parameter k1               |$        0.693$ |second^-1^ |
|Value of parameter k2               |$         0.25$ |second^-1^ |
|Volume of compartment "compartment" |$            1$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

Note: The test data for this model was generated from an analytical solution
of the system of equations.

*)

newcase[ "00344" ];

addCompartment[ compartment, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.5 10^-3 ];
addSpecies[ S2, initialAmount -> 2.0 10^-3 ];
addSpecies[ S3, initialAmount -> 1.5 10^-3, boundaryCondition->True  ];
addSpecies[ S4, initialAmount -> 4 10^-3 ];
addParameter[ k1, value -> .693 ];
addParameter[ k2, value -> .25 ];
addRule[ type->RateRule, variable -> S3, math -> k1  *  0.5 10^-3];
addRule[ type->RateRule, variable -> S4, math -> -k2  *  0.5 10^-3];
addReaction[ S1 -> S2, reversible -> False,
	     kineticLaw -> k1 * S1 * compartment ];

makemodel[]

