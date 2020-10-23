(* 

category:      Test
synopsis:      Model using parameters and rules only with an algebraic rule.
componentTags: Parameter, RateRule, AlgebraicRule
testTags:      NonConstantParameter
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains four variable parameters called p1, p2, p3 and p4 
and two constant parameters called k1 and k2.  The model contains 
four rules defined as:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Algebraic |   n/a    | $p4 + -1 * p1$  |
 | Rate      |   p1     | $-k1 * p1 * p2 + k2 * p3$  |
 | Rate      |   p2     | $-k1 * p1 * p2 + k2 * p3$  |
 | Rate      |   p3     | $k1 * p1 * p2 - k2 * p3$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*                   |
|Initial value of parameter p1       |$1.0 \x 10^-3$  |any                      |
|Initial value of parameter p2       |$2.0 \x 10^-3$  |same as p1                      |
|Initial value of parameter p3       |$1.0 \x 10^-3$  |same as p1                      |
|Initial value of parameter p4       |$1.0 \x 10^-3$  |same as p1                      |
|Value of parameter k1               |$0.75$           |(units of p1)^-1^ second^-1^ |
|Value of parameter k2               |$0.25$           |second^-1^                |]

*)

newcase[ "00576" ];

addParameter[ p1, value -> 1.0 10^-3, constant -> False];
addParameter[ p2, value -> 2.0 10^-3, constant -> False];
addParameter[ p3, value -> 1.0 10^-3, constant -> False];
addParameter[ p4, value -> 1.0 10^-3, constant -> False];
addParameter[ k1, value -> 0.75 ];
addParameter[ k2, value -> 0.25 ];
addRule[ type->AlgebraicRule, math -> p4 - p1];
addRule[ type->RateRule, variable -> p1, math -> -k1 * p1 * p2 + k2 * p3];
addRule[ type->RateRule, variable -> p2, math -> -k1 * p1 * p2 + k2 * p3];
addRule[ type->RateRule, variable -> p3, math -> k1 * p1 * p2 - k2 * p3];

makemodel[]

