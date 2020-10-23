(* 

category:      Test
synopsis:      Model using parameters and rules only.
componentTags: Parameter, RateRule 
testTags:      NonConstantParameter
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains three varying parameters called S1, S2 and S3 and two constant
parameters called k1 and k2.  (Note that indeed S1, S2 and S3 are parameters and 
not species in this model.)

The model contains three rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Rate | S1       | $k2 * S3-k1 * S1 * S2$  |
 | Rate | S2       | $k2 * S3-k1 * S1 * S2$  |
 | Rate | S3       | $k1 * S1 * S2-k2 * S3$  |]


The initial conditions are as follows:

[{width:30em,margin: 1em auto}|      |*Value*          |*Units*  |
|Value of parameter S1 |$1.5 \x 10^-2$ |any                        |
|Value of parameter S2 |$  2 \x 10^-2$ |same as S1                 |
|Value of parameter S3 |$  1 \x 10^-2$ |same as S1                 |
|Value of parameter k1 |$          0.5$ |(units of S1)^-1^ second^-1^ |
|Value of parameter k2 |$          0.5$ |second^-1^                 |]


*)

newcase[ "00165" ];

addParameter[ S1, value -> 1.5 10^-2, constant -> False ];
addParameter[ S2, value -> 2 10^-2, constant -> False ];
addParameter[ S3, value -> 1 10^-2, constant -> False ];
addParameter[ k1, value -> 0.5 ];
addParameter[ k2, value -> 0.5 ];
addRule[ type->RateRule, variable -> S1, math -> k2 * S3-k1 * S1 * S2];
addRule[ type->RateRule, variable -> S2, math -> k2 * S3-k1 * S1 * S2];
addRule[ type->RateRule, variable -> S3, math -> k1 * S1 * S2-k2 * S3];

makemodel[]

