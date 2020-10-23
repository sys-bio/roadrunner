(*

category:      Test
synopsis:      Three constant parameters, with initialassignments to inf, -inf, and nan.
componentTags: InitialAssignment, Parameter
testTags:      InitialValueReassigned
testType:      TimeCourse
levels:        2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

 This model tests the 'INF' and 'NaN' constructs with initial assignments.

The model contains:
* 3 parameters (P, Q, R)
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter P | $INF$ | constant |
| Initial value of parameter Q | $-INF$ | constant |
| Initial value of parameter R | $NaN$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

