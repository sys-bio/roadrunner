(* 

category:      Test
synopsis:      Model using parameters and rules only with an algebraic rule.
componentTags: Parameter, RateRule, AlgebraicRule
testTags:      NonConstantParameter, InitialValueReassigned
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains four variable parameters called p1, p2, p3 and p4 
and three constant parameters called k1, k2 and k3.  The model contains 
five rules defined as:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Algebraic |   n/a    | $k2 - 0.9$  |]
 | Rate      |   p1     | $-k1 * p1 * p2 + k2 * p3 + k3 * p3$  |
 | Rate      |   p2     | $-k1 * p1 * p2 + k2 * p3$  |
 | Rate      |   p3     | $k1 * p1 * p2 - k2 * p3 - k3 * p3$  |
 | Rate      |   p4     | $k3 * p3$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|       |*Value*          |*Units*  |
|Initial value of parameter p1       |$2.0 \x 10^-4$  |any                      |
|Initial value of parameter p2       |$2.0 \x 10^-4$  |same as p1                      |
|Initial value of parameter p3       |$0$              |same as p1                      |
|Initial value of parameter p4       |$0$              |same as p1                      |
|Value of parameter k1               |$1.0 \x 10^4$   |(units of p1)^-1^ second^-1^  |
|Value of parameter k2               |$undeclared$     |second^-1^                 |
|Value of parameter k3               |$0.7$            |second^-1^                 |]

*)

newcase[ "00575" ];

addParameter[ p1, value -> 2.0 10^-4, constant -> False];
addParameter[ p2, value -> 2.0 10^-4, constant -> False];
addParameter[ p3, value -> 0, constant -> False];
addParameter[ p4, value -> 0, constant -> False];
addParameter[ k1, value -> 1.0 10^4 ];
addParameter[ k2, constant -> False ];
addParameter[ k3, value -> 0.7 ];
addRule[ type->AlgebraicRule, math -> k2 - 0.9];
addRule[ type->RateRule, variable -> p1, math -> -k1 * p1 * p2 + k2 * p3 + k3 * p3];
addRule[ type->RateRule, variable -> p2, math -> -k1 * p1 * p2 + k2 * p3];
addRule[ type->RateRule, variable -> p3, math -> k1 * p1 * p2 - k2 * p3 - k3 * p3];
addRule[ type->RateRule, variable -> p4, math -> k3 * p3];

makemodel[]

