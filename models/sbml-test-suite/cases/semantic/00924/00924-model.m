(* 

category:      Test
synopsis:      Model with parameters and assignmentRule only.
componentTags: Parameter, AssignmentRule 
testTags:      InitialValueReassigned
testType:      TimeCourse
levels:        1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

The model contains two constant
parameters called k1 and k2.


The model contains one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| assignmentRule | k1 | $k2 * 3$  |]


Note: AssignmentRules override any declared initial values.  In this
case, the initial value of k1 is inconsistent with the value 
calculated using the assignmentRule.  The calculated value should be used.
.

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|      |*Value*          |*Units*  |
|Value of parameter k1 |$   1.0$ |any |
|Value of parameter k2 |$0.3$ |same as k1 |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.


*)

newcase[ "00924" ];

addParameter[ k1, constant -> False, value->1.0 ];
addParameter[ k2, value->0.3 ];
addRule[ type -> assignmentRule, variable -> k1, math -> k2 * 3];

makemodel[]

