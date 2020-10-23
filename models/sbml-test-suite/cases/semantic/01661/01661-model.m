(*

category:        Test
synopsis:        An event with avogadro in the trigger.
componentTags:   CSymbolAvogadro, CSymbolTime, EventWithDelay, Parameter
testTags:        EventIsNotPersistent, NonConstantParameter
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, the avogadro csymbol is used in an event trigger, and is used in a comparison where it makes a difference whether the event is persistent or not (in this version, it is not).

The model contains:
* 1 parameter (p1)

There is one event:

[{width:40em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Persistent*  |  *Delay*  | *Event Assignments* |
| _E0 | $(time >= 0.5) && (time <= (avogadro / 6.022e23))$ | false | $1$ | $p1 = 3$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p1 | $1$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
