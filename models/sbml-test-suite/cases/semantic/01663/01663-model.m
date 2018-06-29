(*

category:        Test
synopsis:        An event with avogadro in the trigger.
componentTags:   CSymbolAvogadro, EventNoDelay, Parameter
testTags:        EventT0Firing, NonConstantParameter
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, the avogadro csymbol is used in an event trigger, which evaluates to 'true' at t0, and firing only because it's set initialValue=false.

The model contains:
* 1 parameter (p1)

There is one event:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *initialValue*  | *Event Assignments* |
| E0 | $avogadro > 6.022e23$ | false | $p1 = 5$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p1 | $1$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
