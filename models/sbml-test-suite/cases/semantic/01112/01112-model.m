(*

category:      Test
synopsis:      Test of 0-child MathML constructs plus, times, and, or, and xor.
componentTags: InitialAssignment, Parameter
testTags:      InitialValueReassigned, UncommonMathML
testType:      TimeCourse
levels:        2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

In the official MathML spec, plus, times, and, or, and xor may all have 0 arguments.  When they do, they are supposed return identity for that construct, namely:
*plus:  0
*times: 1
*and:   true
*or:    false
*xor:   false

The model contains:
* 5 parameters (a, b, c, d, e)
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter a | $plus()$ | constant |
| Initial value of parameter b | $times()$ | constant |
| Initial value of parameter c | $piecewise(and(), 1, 0)$ | constant |
| Initial value of parameter d | $piecewise(or(), 1, 0)$ | constant |
| Initial value of parameter e | $piecewise(xor(), 1, 0)$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

