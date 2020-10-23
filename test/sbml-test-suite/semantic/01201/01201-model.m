(*

category:        Test
synopsis:        Nested piecewise in conditional and return
componentTags:   AssignmentRule, Parameter, RateRule
testTags:        InitialValueReassigned, NonConstantParameter, UncommonMathML
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 A combination of cases 01199 and 01200, with piecewise functions nested inside a piecewise function as both a conditional and as a return value.  Tests the combination of nesting piecewise functions.

The model contains:
* 3 parameters (z, y, x)

There are 3 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | z | $piecewise(piecewise(2, y > 1.49, 1), piecewise(true, x <= 0.49, false), 0)$ |
| Rate | y | $-2$ |
| Rate | x | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter z | $piecewise(piecewise(2, y > 1.49, 1), piecewise(true, x <= 0.49, false), 0)$ | variable |
| Initial value of parameter y | $2$ | variable |
| Initial value of parameter x | $0$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

