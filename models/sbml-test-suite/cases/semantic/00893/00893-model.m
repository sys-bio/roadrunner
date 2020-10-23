(* 

category:      Test
synopsis:      Model with parameters and rules only, with csymbol time.
componentTags: Parameter, RateRule, CSymbolTime
testTags:      NonConstantParameter
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains three varying parameters called P1, P2 and P3 and one
constant parameter called k1.
 
The model contains three rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Rate | P3       | $k1 * s$  |
 | Rate | P1       | $-(P3 * P1)$  |
 | Rate | P2       | $P3 * P1$  |]
where the symbol 's' denotes the current simulation time.


The initial conditions are as follows:

[{width:30em,margin: 1em auto}|      |*Value*          |*Units*  |
|Value of parameter P1 |$1.5 \x 10^-3$ |any |
|Value of parameter P2 |$            0$ |same as P1 |
|Value of parameter P3 |$  1 \x 10^-3$ |second^-1^ |
|Value of parameter k1 |$  0.5$ |second^-3^ |]


*)

newcase[ "00893" ];

addParameter[ P1, value -> 1.5 10^-3, constant -> False ];
addParameter[ P2, value -> 0, constant -> False ];
addParameter[ P3, value -> 1 10^-3, constant -> False ];
addParameter[ k1, value -> 0.5];
addRule[ type->RateRule, variable -> P3, math -> k1*\[LeftAngleBracket]s, "time"\[RightAngleBracket]];
addRule[ type->RateRule, variable -> P1, math -> -(P3 * P1)];
addRule[ type->RateRule, variable -> P2, math -> P3 * P1];

makemodel[]

