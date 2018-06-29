(*

category:        Test
synopsis:        An event with a delay that has triggers with variable delays.
componentTags:   CSymbolDelay, EventWithDelay, Parameter, RateRule
testTags:        DelayInTrigger, EventIsNotPersistent, NonConstantParameter
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 The model tests whether the delayed value of a parameter is within a window of values, assigning a value to another parameter if so.  The event is not persistent, so even when the tested parameter leaves the window before the event assignment, the assignment is not carried out.

The model contains:
* 2 parameters (P1, P2)

There is one event:

[{width:40em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Persistent*  |  *Delay*  | *Event Assignments* |
| E0 | $(delay(P1, 1) > 1.5) && (delay(P1, 1) < 2)$ | false | $1$ | $P2 = 3$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | P1 | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter P1 | $0$ | variable |
| Initial value of parameter P2 | $0$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
