(*

category:        Test
synopsis:        Nested piecewise in return, conditional, and otherwise.
componentTags:   AssignmentRule, Parameter, RateRule
testTags:        InitialValueReassigned, NonConstantParameter, UncommonMathML
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 The assignment rule for parameter 'z' has three nested piecewise functions in the return, conditional, and otherwise parts of the containing piecewise function.  A combination of 01199, 01200, and 01202, and functionally equivalent to 01203.

The model contains:
* 3 parameters (z, y, x)

There are 3 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | z | $piecewise(piecewise(2, y > 1.49, 1), piecewise(true, x <= 0.49, false), piecewise(1, y > 0.49, 0))$ |
| Rate | y | $-2$ |
| Rate | x | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter z | $piecewise(piecewise(2, y > 1.49, 1), piecewise(true, x <= 0.49, false), piecewise(1, y > 0.49, 0))$ | variable |
| Initial value of parameter y | $2$ | variable |
| Initial value of parameter x | $0$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

