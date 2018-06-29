(*

category:        Test
synopsis:        A rateOf csymbol in an event assignment.
componentTags:   CSymbolRateOf, CSymbolTime, EventNoDelay, Parameter, RateRule
testTags:        NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Analytic
packagesPresent: 

 A rateOf csymbols is used here in an event assignment for p2.

The model contains:
* 2 parameters (p1, p2)

There is one event:

[{width:30em,margin: 1em auto}|  *Event*  |  *Trigger*  | *Event Assignments* |
| _E0 | $time > 5.5$ | $p2 = rateOf(p1)$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | p1 | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p1 | $2$ | variable |
| Initial value of parameter p2 | $10$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
