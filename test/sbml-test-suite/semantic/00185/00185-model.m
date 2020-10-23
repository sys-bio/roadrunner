(* 

category:      Test
synopsis:      Model using parameters and both rate and assignment rules
               including initialAssignments.   
componentTags: Parameter, InitialAssignment, RateRule, AssignmentRule 
testTags:      NonConstantParameter, InitialValueReassigned
testType:      TimeCourse
levels:        2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The model contains three varying parameters called S1, S2 and S3 and two constant
parameters called k1 and k2.  (Note that indeed S1, S2 and S3 are parameters and 
not species in this model.)

The model contains three rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
 | Assignment | S3 | $1 * S2$  |
 | Rate       | S1 | $-k2 * S1 $  |
 | Rate       | S2 | $k2 * S1 $  |]

The model contains two initialAssignments:

[{width:30em,margin: 1em auto}| Variable | Formula |
 | S1 | $k1 / 0.75$  |
 | S2 | $k2 * 3 \x 10^-17$  |]

Note: InitialAssignments override any declared initial values.  The initial
values of parameters S1, S2 and S3 are undeclared and so must be calculated
by the initialAssignments.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|      |*Value*         |*Units*  |
|Value of parameter S1              |$   undeclared$ |any |
|Value of parameter S2              |$   undeclared$ |same as S1 |
|Value of parameter S3              |$   undeclared$ |same as S1 |
|Value of parameter k1              |$         0.75$ |dimensionless |
|Value of parameter k2              |$           50$ |second^-1^ |]

*)

newcase[ "00185" ];

addParameter[ S1, constant -> False ];
addParameter[ S2, constant -> False  ];
addParameter[ S3, constant -> False  ];
addParameter[ k1, value -> 0.75 ];
addParameter[ k2, value -> 50 ];
addInitialAssignment[ S1, math -> k1/0.75];
addInitialAssignment[ S2, math -> k2 * 3 10^-17];
addRule[ type->AssignmentRule, variable -> S3, math ->k1 * S2];
addRule[ type->RateRule, variable -> S1, math -> -k2 * S1 ];
addRule[ type->RateRule, variable -> S2, math -> k2 * S1 ];

makemodel[]

