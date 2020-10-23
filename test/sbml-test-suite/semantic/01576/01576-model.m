(*

category:        Test
synopsis:        A non-persistent event influenced by an algebraic rule.
componentTags:   AlgebraicRule, EventWithDelay, Parameter, RateRule
testTags:        EventIsNotPersistent, InitialValueReassigned, NonConstantParameter
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 The model has a non-persistent event whose trigger is controlled by an algebraic rule such that it does not actually fire, being 'true' for less time than its delay.  This is the same as model 1575, but with a non-persistent event.

The model contains:
* 2 parameters (k1, k2)

There is one event:

[{width:40em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Persistent*  |  *Delay*  | *Event Assignments* |
| E0 | $(k2 > 3.5) && (k2 < 4)$ | false | $1$ | $k1 = 1$ |]


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
