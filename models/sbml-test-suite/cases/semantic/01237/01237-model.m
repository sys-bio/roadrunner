(*

category:        Test
synopsis:        An EventAssignment with missing MathML.
componentTags:   CSymbolTime, EventNoDelay, Parameter
testTags:        NoMathML, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Analytic
packagesPresent: 

This test was written to ensure that if an EventAssignment with no Math child is encountered, it is ignored.

The model contains:
* 1 parameter (p)

There is one event:

[{width:30em,margin: 1em auto}|  *Event*  |  *Trigger*  | *Event Assignments* |
| E0 | $time > 5.5$ | $p = $ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p | $3$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
