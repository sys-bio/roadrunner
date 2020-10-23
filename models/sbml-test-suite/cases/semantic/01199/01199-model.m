(*

category:        Test
synopsis:        Nested piecewise in conditional.
componentTags:   AssignmentRule, Parameter, RateRule
testTags:        InitialValueReassigned, NonConstantParameter, UncommonMathML
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 The assignment rule for parameter 'z' has a nested piecewise function in the conditional part of the piecewise function, that returns 'true' or 'false'.  It is functionally equivalent to simply having the nested conditional there directly, and is there merely to test the presence of the nesting itself.

The model contains:
* 2 parameters (x, z)

There are 2 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | x | $1$ |
| Assignment | z | $piecewise(1, piecewise(true, x >= 0.49, false), 0)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter x | $0$ | variable |
| Initial value of parameter z | $piecewise(1, piecewise(true, geq(x, 0.49), false), 0)$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

