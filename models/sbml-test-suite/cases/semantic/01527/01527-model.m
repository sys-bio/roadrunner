(*

category:        Test
synopsis:        An event that fires at t0 because of the rate of a parameter.
componentTags:   CSymbolRateOf, EventNoDelay, Parameter, RateRule
testTags:        EventT0Firing, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, an event fires at t0 because the rate of P1 is greater than 0.5.

The model contains:
* 2 parameters (P1, P2)

There is one event:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *initialValue*  | *Event Assignments* |
| E0 | $rateOf(P1) > 0.5$ | false | $P2 = 3$ |]


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
