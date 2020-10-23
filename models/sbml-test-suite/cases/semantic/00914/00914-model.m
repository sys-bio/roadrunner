(* 

category:      Test
synopsis:      Model varying compartment using rules only.
componentTags: Compartment, RateRule, FunctionDefinition 
testTags:      NonConstantCompartment, NonUnityCompartment
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

The model contains one varying 2-dimensional compartment called c.

The model contains one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*         |
 | Rate                                 | c           | $multiply(0.5, c)$  |]

The model contains one functionDefinition defined as:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | multiply | x, y | $x * y$ |]


The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value*         |*Units*  |
|Size of compartment c         |$0.5$ |metre^2^        |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.


*)

newcase[ "00914" ];

addFunction[ multiply, arguments -> {x, y}, math -> x * y];
addCompartment[ c, size -> 0.5, spatialDimensions -> 2, constant -> False ];
addRule[ type->RateRule, variable -> c, math -> multiply[c, 0.5]];

makemodel[]

