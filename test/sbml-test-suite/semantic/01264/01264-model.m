(*

category:        Test
synopsis:        A variable rateOf csymbol in a rate rule.
componentTags:   CSymbolRateOf, Parameter, RateRule
testTags:        NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Numeric
packagesPresent: 

 A variable rateOf csymbols is used here in a rate rule, defining the rate of change of p2 the same as the rate of change of p1.

The model contains:
* 2 parameters (p1, p2)

There are 2 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | p1 | $0.01 * p1$ |
| Rate | p2 | $rateOf(p1)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p1 | $2$ | variable |
| Initial value of parameter p2 | $10$ | variable |]

*)
