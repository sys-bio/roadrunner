(*

category:        Test
synopsis:        Nested piecewise in return.
componentTags:   AssignmentRule, Parameter, RateRule
testTags:        InitialValueReassigned, NonConstantParameter, UncommonMathML
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 This model has a nested 'piecewise' function, with the nested piecewise in the 'returned value' part of the function.

The model contains:
* 3 parameters (z, y, x)

There are 3 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | z | $piecewise(piecewise(2, gt(y, 1.49), 1), leq(x, 0.49), 0)$ |
| Rate | y | $-2$ |
| Rate | x | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter z | $piecewise(piecewise(2, gt(y, 1.49), 1), leq(x, 0.49), 0)$ | variable |
| Initial value of parameter y | $2$ | variable |
| Initial value of parameter x | $0$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

