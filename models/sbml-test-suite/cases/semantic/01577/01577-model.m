(*

category:        Test
synopsis:        Two event priorities affected by an algebraic rule.
componentTags:   AlgebraicRule, CSymbolTime, EventNoDelay, EventPriority, Parameter, RateRule
testTags:        InitialValueReassigned, NonConstantParameter
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 The model has two competing events whose priorities are set via an algebraic rule.

The model contains:
* 3 parameters (k1, P1, k2)

There are 2 events:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  | *Event Assignments* |
| E0 | $time > 4.5$ | $k1$ | $P1 = 3$ |
| E1 | $time > 4.5$ | $k2$ | $P1 = 5$ |]


There are 2 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Algebraic | $0$ | $10 - k1 - k2$ |
| Rate | k1 | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter k1 | $0$ | variable |
| Initial value of parameter P1 | $1$ | variable |
| Initial value of parameter k2 | $unknown$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
