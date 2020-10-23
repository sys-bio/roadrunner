(* 

category:      Test
synopsis:      Model using rules and parameters with a rate that causes a discontinuity
               in the output.
componentTags: Parameter, RateRule 
testTags:      NonConstantParameter
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains two varying parameters called S1 and S2 and two constant
parameters called p1 and p2.  (Note that indeed S1 and S2 are parameters and 
not species in this model.)

The model contains two rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Rate | S1 | $-Ceiling(S1 * p1)! / p2$  |
 | Rate | S2 | $Ceiling(S1 * p1)! / p2$  |]


The initial conditions are as follows:

[{width:30em,margin: 1em auto}|      |*Value*          |*Units*  |
|Value of parameter S1 |$          1.0$ |any |
|Value of parameter S2 |$          0.0$ |same as S1 |
|Value of parameter p1 |$             $ |(units of S1)^-1^ |
|Value of parameter p2 |$            2$ |second (units of S1)^-1^ |]


*)

newcase[ "00173" ];

addParameter[ S1, value -> 1.0, constant -> False ];
addParameter[ S2, value -> 0.0, constant -> False ];
addParameter[ p1, value -> 4];
addParameter[ p2, value -> 25];
addRule[ type->RateRule, variable -> S1, math -> -Ceiling[S1 * p1]!/p2];
addRule[ type->RateRule, variable -> S2, math -> Ceiling[S1 * p1]!/p2];

makemodel[]

