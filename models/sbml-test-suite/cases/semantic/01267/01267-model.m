(*

category:        Test
synopsis:        A variable rateOf csymbol in an event priority.
componentTags:   CSymbolRateOf, CSymbolTime, EventNoDelay, EventPriority, Parameter, RateRule
testTags:        NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Analytic
packagesPresent: 

 A variable rateOf csymbol is used here in an event priority that assigns to p2.  At the time the event triggers, the priority is greater than .021, meaning that it fires first, and E0 fires second, leaving the value of p2 at 5.

The model contains:
* 2 parameters (p1, p2)

There are 2 events:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  | *Event Assignments* |
| E0 | $time > 5.5$ | $0.021$ | $p2 = 5$ |
| E1 | $time > 5.5$ | $rateOf(p1)$ | $p2 = 3$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | p1 | $0.01 * p1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p1 | $2$ | variable |
| Initial value of parameter p2 | $10$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
