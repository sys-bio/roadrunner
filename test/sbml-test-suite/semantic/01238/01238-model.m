(*

category:        Test
synopsis:        An Event Trigger with missing MathML.
componentTags:   EventNoDelay, Parameter
testTags:        NoMathML, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Analytic
packagesPresent: 

This test was written to ensure that if an Event Trigger with no Math child is encountered, the Event is ignored.

The model contains:
* 1 parameter (p)

There is one event:

[{width:30em,margin: 1em auto}|  *Event*  |  *Trigger*  | *Event Assignments* |
| E0 | $(unset)$ | $p = 2$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p | $3$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
