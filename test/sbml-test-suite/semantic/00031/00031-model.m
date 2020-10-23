(* 

category:      Test
synopsis:      Basic single rate rule used to determine species value.
componentTags: Compartment, Species, RateRule 
testTags:      Amount
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

The model contains one compartment called "compartment".  There is one
species named S1.  The model does not contain any reactions.  The model
contains one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | S1       | $7$     |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*         |*Units*  |
|Initial amount of S1                |$            0$ |mole                      |
|Volume of compartment "compartment" |$            1$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

newcase[ "00031" ];

addCompartment[ compartment, size -> 1.0];
addSpecies[ S1, initialAmount -> 0 ];
addRule[ type->RateRule, variable -> S1, math -> 7];

makemodel[]

