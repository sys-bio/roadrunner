(* 

category:      Test
synopsis:      Two rate rules used to determine species value.
componentTags: Compartment, Species, Parameter, RateRule 
testTags:      Amount
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

The model contains one compartment called "compartment".  There are two
species named S1 and S2 and one parameter named k1.  The model does not
contain any reactions.  The model contains two rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | S1       | $-(k1 * S1)$  |
| Rate | S2       | $k1 * S1$     |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*         |*Units*  |
|Initial amount of S1                |$1.5 \x 10^-2$ |mole                      |
|Initial amount of S2                |$            0$ |mole                      |
|Value of parameter k1               |$            1$ |second^-1^                |
|Volume of compartment "compartment" |$            1$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

newcase[ "00032" ];

addCompartment[ compartment, size -> 1 ];
addSpecies[ S1, initialAmount -> 1.5 10^-2 ];
addSpecies[ S2, initialAmount -> 0 ];
addParameter[ k1, value -> 1 ];
addRule[ type->RateRule, variable -> S1, math -> -(k1 * S1)];
addRule[ type->RateRule, variable -> S2, math -> k1 * S1];

makemodel[]

