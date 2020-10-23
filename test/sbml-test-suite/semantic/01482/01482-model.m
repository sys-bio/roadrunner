(*

category:        Test
synopsis:        An algebraic rule with a 'rateOf' csymbol.
componentTags:   AlgebraicRule, CSymbolRateOf, Parameter, RateRule
testTags:        InitialValueReassigned, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, we have an algebraic rule that uses the 'rateOf' function.

The model contains:
* 3 parameters (S1, x, P1)

There are 2 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | S1 | $2$ |
| Algebraic | $0$ | $rateOf(S1) - P1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter S1 | $1$ | variable |
| Initial value of parameter x | $unknown$ | variable |
| Initial value of parameter P1 | $unknown$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
