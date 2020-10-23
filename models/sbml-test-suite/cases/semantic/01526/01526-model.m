(*

category:        Test
synopsis:        A delayed event with a trigger window for the rateOf another parameter.
componentTags:   CSymbolRateOf, CSymbolTime, EventWithDelay, Parameter, RateRule
testTags:        EventIsNotPersistent, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, a delayed event triggers based on whether the value of the rate of change of a parameter is within a window.  The event is not persistent, meaning that it will not assign its value because the rate of change goes outside the window before execution.

The model contains:
* 2 parameters (P1, P2)

There is one event:

[{width:40em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Persistent*  |  *Delay*  | *Event Assignments* |
| E0 | $(rateOf(P1) > 1.5) && (rateOf(P1) < 2.5)$ | false | $2$ | $P2 = 3$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | P1 | $time$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter P1 | $0$ | variable |
| Initial value of parameter P2 | $0$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
