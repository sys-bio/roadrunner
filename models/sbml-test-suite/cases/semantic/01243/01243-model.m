(*

category:        Test
synopsis:        Two EventAssignments, one with missing MathML.
componentTags:   CSymbolTime, EventNoDelay, Parameter
testTags:        NoMathML, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Analytic
packagesPresent: 

This test was written to ensure that even if one EventAssignment is to be ignored due to it having missing MathML, there may be other EventAssignments that are normal.

The model contains:
* 2 parameters (p, d)

There is one event:

[{width:30em,margin: 1em auto}|  *Event*  |  *Trigger*  | *Event Assignments* |
| _E0 | $time > 5.5$ | $d = $ |
|  |  | $p = 5$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p | $3$ | variable |
| Initial value of parameter d | $5$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
