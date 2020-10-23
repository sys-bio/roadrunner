(* 

category:      Test
synopsis:      Model varying compartment using rules only.
componentTags: Compartment, RateRule 
testTags:      NonConstantCompartment, NonUnityCompartment
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

The model contains one varying 1-dimensional compartment called c.  

The model contains one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*         |
 | Rate                                 | c           | $0.1 * c$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value*         |*Units*  |
|Size of compartment c         |$1.5 \x 10^-4$ |metre        |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.


*)

newcase[ "00905" ];

addCompartment[ c, size -> 1.5 10^-4, spatialDimensions -> 1, constant -> False ];
addRule[ type->RateRule, variable -> c, math -> 0.1*c];

makemodel[]

