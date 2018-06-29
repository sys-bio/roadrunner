(*

category:        Test
synopsis:        An event triggered at t0 by a delayed value in its trigger.
componentTags:   AssignmentRule, CSymbolDelay, CSymbolTime, EventNoDelay, Parameter
testTags:        DelayInTrigger, EventT0Firing, InitialValueReassigned, NonConstantParameter
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, an event tests whether the delayed value of a parameter is less than -0.5.  Since that parameter has an assignment rule to be equal to 'time', which applies at all times, including before the beginning of the simulation, that trigger evaluates to true, and fires because it is set to have an initial value of 'false'.

The model contains:
* 2 parameters (P1, P2)

There is one event:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *initialValue*  | *Event Assignments* |
| E0 | $delay(P1, 1) < -0.5$ | false | $P2 = 3$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | P1 | $time$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter P1 | $time$ | variable |
| Initial value of parameter P2 | $0$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
