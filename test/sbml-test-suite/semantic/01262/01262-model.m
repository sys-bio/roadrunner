(*

category:        Test
synopsis:        A rateOf csymbol in an event priority.
componentTags:   CSymbolRateOf, CSymbolTime, EventNoDelay, EventPriority, Parameter, RateRule
testTags:        NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Numeric
packagesPresent: 

 A rateOf csymbol is used here in an event priority that assigns to p2.

The model contains:
* 2 parameters (p1, p2)

There are 2 events:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  | *Event Assignments* |
| E0 | $time > 5.5$ | $1.01$ | $p2 = 5$ |
| E1 | $time > 5.5$ | $rateOf(p1)$ | $p2 = 3$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | p1 | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p1 | $1$ | variable |
| Initial value of parameter p2 | $10$ | variable |]

*)
