(* 

category:      Test
synopsis:      Model using parameters and rules only with an algebraic rule.
componentTags: Parameter, RateRule, AlgebraicRule
testTags:      NonConstantParameter
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains four variable parameters called p1, p2, p3 and p4 
and three constant parameters called k1, k2 and k3.  The model contains 
four rules defined as:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Algebraic |   n/a    | $(1 + k3) * p4 - p3$  |
 | Rate      |   p1     | $-p1 * k1$  |
 | Rate      |   p2     | $p4 * k2$  |
 | Rate      |   p3     | $p1 * k1 - k2 * p4$  |]

The initial conditions are as follows:

[{width:30em,margin: 1em auto}  | |*Value* |*Units*  |
|Initial value of parameter p1 |$1$     |any                      |
|Initial value of parameter p2 |$0$     |same as p1                      |
|Initial value of parameter p3 |$0$     |same as p1                      |
|Initial value of parameter p4 |$0$     |same as p1                      |
|Value of parameter k1         |$0.1$   |second^-1^            |
|Value of parameter k2         |$0.375$ |second^-1^                |
|Value of parameter k3         |$2.5$   |dimensionless                |]


*)

newcase[ "00573" ];

addParameter[ p1, value -> 1, constant -> False];
addParameter[ p2, value -> 0, constant -> False];
addParameter[ p3, value -> 0, constant -> False];
addParameter[ p4, value -> 0, constant -> False];
addParameter[ k1, value -> 0.1 ];
addParameter[ k2, value -> 0.375 ];
addParameter[ k3, value -> 2.5 ];
addRule[ type->AlgebraicRule, math -> (1+k3) * p4 - p3];
addRule[ type->RateRule, variable -> p1, math -> -p1 * k1];
addRule[ type->RateRule, variable -> p3, math -> p1 * k1 - k2 * p4];
addRule[ type->RateRule, variable -> p2, math -> p4 * k2];

makemodel[]

