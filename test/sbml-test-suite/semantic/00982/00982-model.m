(*

category:      Test
synopsis:      An assignment rule with a continually-changing delay. 
componentTags: AssignmentRule, CSymbolDelay, CSymbolTime, Parameter, RateRule
testTags:      NonConstantParameter, InitialValueReassigned
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

 This model contains a single parameter x changing at a constant rate, and a second parameter y with a continually-changing delay for an assignment rule:  it takes the value of x at 'temp' time ago, which in turn is assigned the value of time/2.  Effectively, this means that y changes at half the rate of x.

The model contains:
* 3 parameters (x, temp, y)

There are 3 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | x | $1$ |
| Assignment | temp | $time / 2$ |
| Assignment | y | $delay(x, temp)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter x | $1$ | variable |
| Initial value of parameter temp | $time / 2$ | variable |
| Initial value of parameter y | $delay(x, temp)$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

