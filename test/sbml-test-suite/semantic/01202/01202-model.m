(*

category:        Test
synopsis:        Nested piecewise in otherwise.
componentTags:   AssignmentRule, Parameter, RateRule
testTags:        InitialValueReassigned, NonConstantParameter, UncommonMathML
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 The assignment rule for parameter 'z' has a nested piecewise function in the 'otherwise' part of the piecewise function.

The model contains:
* 3 parameters (z, x, y)

There are 3 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | z | $piecewise(2, x <= 0.49, piecewise(1, y > 0.49, 0))$ |
| Rate | x | $1$ |
| Rate | y | $-2$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter z | $piecewise(2, x <= 0.49, piecewise(1, y > 0.49, 0))$ | variable |
| Initial value of parameter x | $0$ | variable |
| Initial value of parameter y | $2$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

