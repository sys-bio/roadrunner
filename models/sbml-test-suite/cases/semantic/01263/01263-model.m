(*

category:        Test
synopsis:        A rateOf csymbol in an event delay.
componentTags:   CSymbolRateOf, CSymbolTime, EventWithDelay, Parameter, RateRule
testTags:        NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Analytic
packagesPresent: 

 A rateOf csymbol is used here in an event delay that assigns to p2.

The model contains:
* 2 parameters (p1, p2)

There is one event:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Delay*  | *Event Assignments* |
| E0 | $time > 4.5$ | $rateOf(p1)$ | $p2 = 5$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | p1 | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p1 | $1$ | variable |
| Initial value of parameter p2 | $10$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
