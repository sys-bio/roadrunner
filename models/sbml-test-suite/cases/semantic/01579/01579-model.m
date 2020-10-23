(*

category:        Test
synopsis:        An event whose delayed assignment is dependent on an algebraic rule.
componentTags:   AlgebraicRule, EventWithDelay, Parameter, RateRule
testTags:        EventUsesAssignmentTimeValues, InitialValueReassigned, NonConstantParameter
testType:        TimeCourse
levels:          2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 The model has a delayed event which assigns the assignment-time value of one parameter to another.  The first is under control of an algebraic rule.

The model contains:
* 3 parameters (k1, P1, k2)

There is one event:

[{width:40em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Use values from:*  |  *Delay*  | *Event Assignments* |
| E0 | $k1 > 4.5$ | Assignment time | $2$ | $P1 = k2$ |]


There are 2 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | k1 | $1$ |
| Algebraic | $0$ | $k1 - k2$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter k1 | $0$ | variable |
| Initial value of parameter P1 | $1$ | variable |
| Initial value of parameter k2 | $unknown$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
