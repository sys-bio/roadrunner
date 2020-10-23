(*

category:      Test
synopsis:      An assignment rule with a continually-changing delay.
componentTags: AssignmentRule, CSymbolDelay, Parameter, RateRule
testTags:      NonConstantParameter, InitialValueReassigned
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

This model contains a single parameter x changing at a constant rate, and a second parameter z that follows x with a delay of 1, and a third parameter y that follows x with a delay of z.

The model contains:
* 3 parameters (z, x, y)

There are 3 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | z | $delay(x, 1)$ |
| Rate | x | $1$ |
| Assignment | y | $delay(x, z)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter z | $delay(x, 1)$ | variable |
| Initial value of parameter x | $1$ | variable |
| Initial value of parameter y | $delay(x, z)$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

