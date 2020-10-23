(* 

category:      Test
synopsis:      Model with parameters and rules with a functionDefinition.
componentTags: Parameter, FunctionDefinition, RateRule 
testTags:      NonConstantParameter
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains three varying parameters called S1, S2 and S3 and two constant
parameters called k1 and k2.  (Note that indeed S1, S2 and S3 are parameters and 
not species in this model.)
 
The model contains three rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Rate | S3       | $multiply(k1, S1, S2)$  |
 | Rate | S1       | $k2 * S3 - multiply(k1, S1, S2)$  |
 | Rate | S2       | $k2 * S3 - multiply(k1, S1, S2)$  |]


The model contains one functionDefinition defined as:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | multiply | x, y, z | $x * y * z$ |]


The initial conditions are as follows:

[{width:30em,margin: 1em auto}|      |*Value*          |*Units*  |
|Value of parameter S1 |$1.0 \x 10^-1$ |any |
|Value of parameter S2 |$2.0 \x 10^-1$ |same as S1 |
|Value of parameter S3 |$1.0 \x 10^-1$ |same as S1 |
|Value of parameter k1 |$         0.75$ |(units of S1)^-1^ second^-1^ |
|Value of parameter k2 |$         0.25$ |second^-1^ |]


*)

newcase[ "00178" ];

addFunction[ multiply, arguments -> {x, y, z}, math -> x * y * z];
addParameter[ S1, value -> 1.0 10^-1, constant -> False ];
addParameter[ S2, value -> 2.0 10^-1, constant -> False ];
addParameter[ S3, value -> 1.0 10^-1, constant -> False ];
addParameter[ k1, value -> 0.75 ];
addParameter[ k2, value -> 0.25 ];
addRule[ type->RateRule, variable -> S3, math -> multiply[k1, S1, S2] - k2 * S3];
addRule[ type->RateRule, variable -> S1, math -> k2 * S3 - multiply[k1, S1, S2]];
addRule[ type->RateRule, variable -> S2, math -> k2 * S3 - multiply[k1, S1, S2]];

makemodel[]

