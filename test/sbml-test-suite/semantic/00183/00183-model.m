(* 

category:      Test
synopsis:      Model using parameters and both rate and assignment rules
               with a functionDefinition.   
componentTags: Parameter, FunctionDefinition, RateRule, AssignmentRule 
testTags:      NonConstantParameter, InitialValueReassigned
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains three varying parameters called S1, S2 and S3 and two constant
parameters called k1 and k2.  (Note that indeed S1, S2 and S3 are parameters and 
not species in this model.)

The model contains three rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Assignment | S3       | $multiply(k1, S2)$  |
 | Rate       | S1       | $-multiply(k2, S1)$  |
 | Rate       | S2       | $k2 * S1 $  |]

The model contains one functionDefinition defined as:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | multiply | x, y | $x * y$ |]

Note the initial value for parameter S3 is undeclared and must be
calculated using the assignmentRule.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|      |*Value*          |*Units*  |
|Value of parameter S1 |$            1$ |any |
|Value of parameter S2 |$          0.5$ |same as S1 |
|Value of parameter S3 |$   undeclared$ |any |
|Value of parameter k1 |$          0.7$ |(units of S3) (units of S1)^-1^ |
|Value of parameter k2 |$           40$ |second^-1^ |]


*)

newcase[ "00183" ];

addFunction[ multiply, arguments -> {x, y}, math -> x * y];
addParameter[ S1, value -> 1, constant -> False ];
addParameter[ S2, value -> 0.5, constant -> False  ];
addParameter[ S3, constant -> False  ];
addParameter[ k1, value -> 0.7 ];
addParameter[ k2, value -> 40 ];
addRule[ type->AssignmentRule, variable -> S3, math ->multiply[k1,S2]];
addRule[ type->RateRule, variable -> S1, math -> -multiply[k2,S1] ];
addRule[ type->RateRule, variable -> S2, math -> k2 * S1 ];

makemodel[]

