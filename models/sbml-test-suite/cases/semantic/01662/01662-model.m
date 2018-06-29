(*

category:        Test
synopsis:        An event with avogadro in the priority.
componentTags:   CSymbolAvogadro, CSymbolTime, EventNoDelay, EventPriority, Parameter
testTags:        NonConstantParameter
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, the avogadro csymbol is used in an event priority, determining which of two simultaneous events fires first.

The model contains:
* 1 parameter (p1)

There are 2 events:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  | *Event Assignments* |
| E0 | $time >= 2.5$ | $1$ | $p1 = 3$ |
| E1 | $time >= 2.5$ | $avogadro / 6.022e23$ | $p1 = 5$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p1 | $1$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
