(*

category:        Test
synopsis:        An event with avogadro in the delay.
componentTags:   CSymbolAvogadro, CSymbolTime, EventWithDelay, Parameter
testTags:        NonConstantParameter
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, the avogadro csymbol is used in an event delay.

The model contains:
* 1 parameter (p1)

There is one event:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Delay*  | *Event Assignments* |
| _E0 | $time >= 1$ | $avogadro / 6.022e23$ | $p1 = 3$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p1 | $1$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
