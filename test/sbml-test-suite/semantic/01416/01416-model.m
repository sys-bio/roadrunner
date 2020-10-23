(*

category:        Test
synopsis:        A delay function for a variable with a rate rule with no MathML.
componentTags:   AssignmentRule, CSymbolDelay, Parameter, RateRule
testTags:        InitialValueReassigned, NoMathML, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Numeric
packagesPresent: 

 In this model, one parameter shadows another parameter via a delay, but the other parameter's rate rule is empty, meaning it does not change.

The model contains:
* 2 parameters (S1, P1)

There are 2 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | S1 | $$ |
| Assignment | P1 | $delay(S1, 1)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter S1 | $3$ | variable |
| Initial value of parameter P1 | $delay(S1, 1)$ | variable |]

*)
