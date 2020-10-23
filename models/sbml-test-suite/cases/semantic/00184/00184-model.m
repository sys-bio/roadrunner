(* 

category:      Test
synopsis:      Model using parameters and both rate and algebraic rules
               with functionDefinitions.   
componentTags: Parameter, FunctionDefinition, RateRule, AlgebraicRule 
testTags:      NonConstantParameter
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains two varying parameters called S1 and S2 and two constant
parameters called k1 and k2.  (Note that indeed S1 and S2 are parameters and not 
species in this model.)

The model contains two rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Algebraic |          | $add(S1, add(S2, -k1))$  |
 | Rate      | S1       | $multiply(k2, S2)$  |]


The model contains two functionDefinitions defined as:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | multiply | x, y      | $x * y$ |
 | add      | x, y      | $x + y$ |]


The initial conditions are as follows:

[{width:30em,margin: 1em auto}|      |*Value*          |*Units*  |
|Value of parameter S1 |$          0.5$ |any |
|Value of parameter S2 |$          0.5$ |same as S1 |
|Value of parameter k1 |$          1.0$ |same as S1 |
|Value of parameter k2 |$          0.8$ |second^-1^ |]


*)

newcase[ "00184" ];

addFunction[ multiply, arguments -> {x, y}, math -> x * y];
addFunction[ add, arguments -> {x, y}, math -> x + y];
addParameter[ S1, value -> 0.5, constant -> False ];
addParameter[ S2, value -> 0.5, constant -> False  ];
addParameter[ k1, value -> 1.0 ];
addParameter[ k2, value -> 0.8 ];
addRule[ type->AlgebraicRule, math -> add[S1,add[S2,-k1]]];
addRule[ type->RateRule, variable -> S1, math -> multiply[k2,S2] ];

makemodel[]

