(*

category:        Test
synopsis:        The rateOf csymbol for a parameter that changes due to a rate rule.
componentTags:   CSymbolRateOf, InitialAssignment, Parameter, RateRule
testTags:        InitialValueReassigned, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Analytic
packagesPresent: 

The rateOf csymbol is used here in an initial assignment that references a parameter changing due to a rate rule, and thus takes the value of that rate rule.

The model contains:
* 2 parameters (p1, p2)

There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | p1 | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p2 | $rateOf(p1)$ | constant |
| Initial value of parameter p1 | $2$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
