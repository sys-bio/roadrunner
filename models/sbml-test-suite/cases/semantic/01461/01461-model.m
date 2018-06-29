(*

category:        Test
synopsis:        The rateOf csymbol of a parameter with a rate rule with no MathML.
componentTags:   AssignmentRule, CSymbolRateOf, Parameter, RateRule
testTags:        InitialValueReassigned, NoMathML, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Analytic
packagesPresent: 

 In this test, the parameter p has a rate rule with no MathML, meaning it does not change in time.  The rateOf csymbol for that parameter therefore equals zero.

The model contains:
* 2 parameters (x, p)

There are 2 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | x | $rateOf(p)$ |
| Rate | p | $$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter x | $rateOf(p)$ | variable |
| Initial value of parameter p | $2$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
