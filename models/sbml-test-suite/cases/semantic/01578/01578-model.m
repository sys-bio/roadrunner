(*

category:        Test
synopsis:        An event that fires at t0 because of an algebraic rule.
componentTags:   AlgebraicRule, EventNoDelay, Parameter
testTags:        EventT0Firing, InitialValueReassigned, NonConstantParameter
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 The model has an event set 'initialValue=false', allowing it to fire at t0.  The variable in the trigger is set via an algebraic rule.

The model contains:
* 2 parameters (k1, P1)

There is one event:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *initialValue*  | *Event Assignments* |
| E0 | $k1 > 4.5$ | false | $P1 = 3$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Algebraic | $0$ | $10 - k1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter k1 | $0$ | variable |
| Initial value of parameter P1 | $1$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
