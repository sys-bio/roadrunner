(* 

category:      Test
synopsis:      Model using parameters and rules only, with csymbol time.
componentTags: Parameter, RateRule, CSymbolTime
testTags:      NonConstantParameter
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains two varying parameters called P1 and P2 and two constant
parameters called k1 and k2.

The model contains two rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*         |
 | Rate                                 | P1           | $P2 * k2 / exp(time)$  |
 | Rate                                 | P2           | $P1 * k1 / exp(time)$  |]
where the symbol 'time' denotes the current simulation time.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value*         |*Units*  |
|Value of parameter P1         |$1.5 \x 10^-15$ |any        |
|Value of parameter P2         |$            0$ |same as P1 |
|Value of parameter k1         |$            1$ |second^-1^ |
|Value of parameter k2         |$          0.75$ |second^-1^ |]


*)

newcase[ "00892" ];

addParameter[ P1, value -> 1.5, constant -> False ];
addParameter[ P2, value -> 0, constant -> False ];
addParameter[ k1, value -> 1 ];
addParameter[ k2, value -> 0.75 ];
addRule[ type->RateRule, variable -> P1, math -> P2 * k2 /Exp[\[LeftAngleBracket]time, "time"\[RightAngleBracket]]];
addRule[ type->RateRule, variable -> P2, math -> P1 * k1 /Exp[\[LeftAngleBracket]time, "time"\[RightAngleBracket]]];

makemodel[]

