(* 

category:      Test
synopsis:      Model using rules and parameters only.
componentTags: Parameter, RateRule 
testTags:      NonConstantParameter
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains one varying parameter called S1.  (Note that indeed 
S1 is a parameter and not species in this model.)

The model contains one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Rate | S1 | $7$  |]


The initial conditions are as follows:

[{width:30em,margin: 1em auto}|      |*Value*          |*Units*  |
|Value of parameter S1 |$          0.0$ |any |]


*)

newcase[ "00175" ];

addParameter[ S1, value -> 0.0, constant -> False ];
addRule[ type->RateRule, variable -> S1, math -> 7];

makemodel[]

