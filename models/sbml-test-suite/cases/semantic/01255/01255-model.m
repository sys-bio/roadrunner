(*

category:        Test
synopsis:        The rateOf csymbol for a parameter whose rate of change is not constant.
componentTags:   AssignmentRule, CSymbolRateOf, Parameter, RateRule
testTags:        InitialValueReassigned, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Analytic
packagesPresent: 

 The rateOf csymbol is used here in an assignment rule that references a parameter changing due to a non-constant rate rule.

The model contains:
* 2 parameters (p1, p2)

There are 2 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | p1 | $0.01 * p1$ |
| Assignment | p2 | $rateOf(p1)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p1 | $1$ | variable |
| Initial value of parameter p2 | $rateOf(p1)$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
