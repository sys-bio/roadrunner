(*

category:        Test
synopsis:        An event with avogadro in an event assignment.
componentTags:   CSymbolAvogadro, CSymbolTime, EventNoDelay, Parameter
testTags:        NonConstantParameter
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, the avogadro csymbol is used as the assignment value in an event.

The model contains:
* 1 parameter (p1)

There is one event:

[{width:30em,margin: 1em auto}|  *Event*  |  *Trigger*  | *Event Assignments* |
| E0 | $time > 1.5$ | $p1 = avogadro$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p1 | $1$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
