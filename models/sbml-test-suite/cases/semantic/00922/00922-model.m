(* 

category:      Test
synopsis:      Model with parameters and initialAssignments only.
componentTags: Parameter, InitialAssignment 
testTags:      
testType:      TimeCourse
levels:        2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

The model contains two constant
parameters called k1 and k2.


The model contains one initialAssignment:

[{width:30em,margin: 1em auto}| Variable | Formula |
 | k1 | $k2 * 2.0$  |]

Note: InitialAssignments override any declared initial values.  In this
case, the initial value of k1 is consistent with the value 
calculated using the initialAssignment. 

The initial conditions are as follows:

[{width:30em,margin: 1em auto}|      |*Value*          |*Units*  |
|Value of parameter k1 |$   0.6$ |any |
|Value of parameter k2 |$0.3$ |same as k1 |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.


*)

newcase[ "00922" ];

addParameter[ k1, constant -> False, value->0.6 ];
addParameter[ k2, value->0.3 ];
addInitialAssignment[ k1, math -> k2 * 2.0];

makemodel[]

