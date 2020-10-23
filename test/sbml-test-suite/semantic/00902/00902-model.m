(* 

category:      Test
synopsis:      Model varying compartment using rules only.
componentTags: Compartment, RateRule 
testTags:      NonConstantCompartment, NonUnityCompartment
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

The model contains one varying compartment called c.  (Note that 
indeed c is a compartment and not species in this model.)

The model contains one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*         |
 | Rate                                 | c           | $0.5 * c$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value*         |*Units*  |
|Size of compartment c         |$1.5 \x 10^-2$ |litre        |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.


*)

newcase[ "00902" ];

addCompartment[ c, size -> 1.5 10^-2, constant -> False ];
addRule[ type->RateRule, variable -> c, math -> c * 0.5];

makemodel[]

