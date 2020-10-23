(* 

category:      Test
synopsis:      Model varying compartment using rules only.
componentTags: Compartment, RateRule 
testTags:      NonConstantCompartment, NonUnityCompartment, MultiCompartment
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

The model contains two varying compartments called C1 and C2, where C2 
is 2-dimensional.

The model contains two rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Rate | C1       | $C2 * 0.15 - C1 * 0.2$  |
 | Rate | C2       | $C1 * 0.3 - C2 * 0.1$  |]


The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value*   |*Units*  |
|Size of compartment C1        |$1.5$     |litre        |
|Size of compartment C2        |$0.75$     |metre^2^ |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.


*)

newcase[ "00908" ];

addCompartment[ C1, size -> 1.5, constant -> False ];
addCompartment[ C2, size -> 0.75, spatialDimensions -> 2, constant -> False ];
addRule[ type->RateRule, variable -> C1, math -> C2 * 0.15-C1 * 0.2];
addRule[ type->RateRule, variable -> C2, math -> C1 * 0.3-C2 * 0.1];

makemodel[]

