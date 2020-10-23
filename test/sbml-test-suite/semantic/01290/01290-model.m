(*

category:        Test
synopsis:        Use of a Boolean in rate rules.
componentTags:   Parameter, RateRule
testTags:        BoolNumericSwap, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Analytic
packagesPresent: 

 The two rate rules have values of 'true' and 'false', yielding a rate of 1 for the first, and 0 for the second.

The model contains:
* 2 parameters (p1, p2)

There are 2 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | p1 | $true$ |
| Rate | p2 | $false$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p1 | $1$ | variable |
| Initial value of parameter p2 | $2$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
