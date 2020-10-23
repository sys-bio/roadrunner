(*

category:        Test
synopsis:        A rateOf csymbol in an event trigger.
componentTags:   AssignmentRule, CSymbolRateOf, EventNoDelay, Parameter, RateRule
testTags:        InitialValueReassigned, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Numeric
packagesPresent: 

 A rateOf csymbol is used here in an event trigger that assigns to p2.

The model contains:
* 3 parameters (p1, p3, p2)

There is one event:

[{width:30em,margin: 1em auto}|  *Event*  |  *Trigger*  | *Event Assignments* |
| _E0 | $rateOf(p1) > 0.0105$ | $p2 = 5$ |]


There are 2 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | p1 | $0.01 * p1$ |
| Assignment | p3 | $rateOf(p1)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p1 | $1$ | variable |
| Initial value of parameter p3 | $rateOf(p1)$ | variable |
| Initial value of parameter p2 | $10$ | variable |]

*)
