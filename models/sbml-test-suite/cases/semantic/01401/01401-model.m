(*

category:        Test
synopsis:        A delay of a rateOf.
componentTags:   AssignmentRule, CSymbolDelay, CSymbolRateOf, Parameter, RateRule
testTags:        InitialValueReassigned, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Numeric
packagesPresent: 

In this model, we test the calculation of the delay of the result of a 'rateOf' csymbol.

The model contains:
* 3 parameters (P0, S1, P1)

There are 3 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | P0 | $rateOf(S1)$ |
| Rate | S1 | $-0.1 * S1$ |
| Assignment | P1 | $delay(rateOf(S1), 1)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter P0 | $rateOf(S1)$ | variable |
| Initial value of parameter S1 | $5$ | variable |
| Initial value of parameter P1 | $delay(rateOf(S1), 1)$ | variable |]

*)
