(* 

category:      Test
synopsis:      Model using parameters and rules only.
componentTags: Parameter, RateRule 
testTags:      NonConstantParameter
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains four varying parameters called S1, S2, S3 and S4 and 
three constant parameters called k1, k2 and k3.  (Note that indeed S1, 
S2, S3, and S4 are parameters and not species in this model.)

The model contains four rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Rate | S1       | $k2 * S3-k1 * S1 * S2 + k3 * S3$  |
 | Rate | S2       | $k2 * S3-k1 * S1 * S2$  |
 | Rate | S3       | $k1 * S1 * S2-k2 * S3-k3 * S3$  |
 | Rate | S4       | $k3 * S3$  |]


The initial conditions are as follows:

[{width:30em,margin: 1em auto}|      |*Value*          |*Units*  |
|Value of parameter S1 |$2.0 \x 10^-2$ |any |
|Value of parameter S2 |$2.0 \x 10^-2$ |same as S1 |
|Value of parameter S3 |$            0$ |same as S1 |
|Value of parameter S4 |$            0$ |same as S1 |
|Value of parameter k1 |$ 1.0 \x 10^2$ |(units of S1)^-1^ second^-1^ |
|Value of parameter k2 |$          0.9$ |second^-1^ |
|Value of parameter k3 |$          0.7$ |second^-1^ |]


*)

newcase[ "00168" ];

addParameter[ S1, value -> 2.0 10^-2, constant -> False];
addParameter[ S2, value -> 2.0 10^-2, constant -> False];
addParameter[ S3, value -> 0, constant -> False];
addParameter[ S4, value -> 0, constant -> False];
addParameter[ k1, value -> 1.0 10^2];
addParameter[ k2, value -> 0.9];
addParameter[ k3, value -> 0.7];
addRule[ type->RateRule, variable -> S1, math -> k2 * S3-k1 * S1 * S2 + k3 * S3];
addRule[ type->RateRule, variable -> S2, math -> k2 * S3-k1 * S1 * S2];
addRule[ type->RateRule, variable -> S3, math -> k1 * S1 * S2-k2 * S3-k3 * S3];
addRule[ type->RateRule, variable -> S4, math -> k3 * S3];

makemodel[]

