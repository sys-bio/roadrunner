(*

category:        Test
synopsis:        An event that assigns the rate of change of a parameter.
componentTags:   CSymbolRateOf, CSymbolTime, EventWithDelay, Parameter, RateRule
testTags:        EventUsesTriggerTimeValues, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, a delayed event assigns the rate of change of a parameter, from the value it had at the trigger time of the event.

The model contains:
* 2 parameters (P1, P2)

There is one event:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Delay*  | *Event Assignments* |
| E0 | $time > 1.5$ | $2$ | $P2 = rateOf(P1)$ |]


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
