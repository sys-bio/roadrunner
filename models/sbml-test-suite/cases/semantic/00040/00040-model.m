(* 

category:      Test
synopsis:      Model using an algebraic rule with a rate rule to determine 
species values.
componentTags: Compartment, Species, Parameter, AlgebraicRule, RateRule 
testTags:      Amount
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one compartment called "compartment".  There are two
species named S1 and S2 and two parameters named k1 and k2.  The model does
not contain any reactions.  The model contains two rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Rate      | S1       | $S2 * k1$  |
 | Algebraic | n/a      | $S1 + S2-k2$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*         |*Units*  |
|Initial amount of S1                |$          0.5$ |mole                      |
|Initial amount of S2                |$          0.5$ |mole                      |
|Value of parameter k1               |$            1$ |second^-1^                |
|Value of parameter k2               |$            1$ |mole litre^-1^            |
|Volume of compartment "compartment" |$            1$ |litre                     |]

The species values are given as amounts of substance to make it easier to
use the model in a discrete stochastic simulator, but (as per usual SBML
principles) their symbols represent their values in concentration units
where they appear in expressions.

*)

newcase[ "00040" ];

addCompartment[ compartment, size -> 1 ];
addSpecies[ S1, initialAmount -> 0.5];
addSpecies[ S2, initialAmount -> 0.5];
addParameter[ k1, value -> 1, constant->True ];
addParameter[ k2, value -> 1, constant->True ];
addRule[ type->RateRule, variable -> S1, math -> S2 * k1];
addRule[ type->AlgebraicRule, math -> S1 + S2-k2];

makemodel[]

