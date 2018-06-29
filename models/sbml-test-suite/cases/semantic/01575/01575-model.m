(*

category:        Test
synopsis:        A persistent event influenced by an algebraic rule.
componentTags:   AlgebraicRule, EventWithDelay, Parameter, RateRule
testTags:        EventIsPersistent, InitialValueReassigned, NonConstantParameter
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, a persistent event has a delay that lasts longer than the trigger is true, so it fires anyway.  It affects the value of a different parameter through an algebraic rule.

The model contains:
* 2 parameters (k1, k2)

There is one event:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Delay*  | *Event Assignments* |
| E0 | $(k2 > 3.5) && (k2 < 4)$ | $1.1$ | $k1 = 1$ |]


There are 2 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Algebraic | $0$ | $k1 - k2$ |
| Rate | k1 | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter k1 | $1$ | variable |
| Initial value of parameter k2 | $unknown$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
