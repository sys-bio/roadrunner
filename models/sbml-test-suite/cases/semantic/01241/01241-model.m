(*

category:        Test
synopsis:        An Event Delay with missing MathML.
componentTags:   CSymbolTime, EventWithDelay, Parameter
testTags:        NoMathML, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Analytic
packagesPresent: 

This test was written to ensure that if an Event Delay with no MathML is encountered, the Delay is ignored.

The model contains:
* 1 parameter (p)

There is one event:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Delay*  | *Event Assignments* |
| _E0 | $time > 5.5$ | $0$ | $p = 5$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p | $3$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
