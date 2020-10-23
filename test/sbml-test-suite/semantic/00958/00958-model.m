(*

category:      Test
synopsis:      Several parameters with initial assignments, testing various L2v1 built-in functions acting on constants.
componentTags: InitialAssignment, Parameter
testTags:      InitialValueReassigned, UncommonMathML
testType:      TimeCourse
levels:        2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

 The model tests the various mathematical constructs added in L2v1, as initial assignments.  As such, the results are identical to case 957.

The model contains:
* 28 parameters (P1, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20, P21, P22, P23, P24, P25, P26, P27, P28, P29)
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter P1 | $pi$ | constant |
| Initial value of parameter P3 | $piecewise(2, gt(1, 0.5), 3)$ | constant |
| Initial value of parameter P4 | $piecewise(2, geq(1, 0.5), 3)$ | constant |
| Initial value of parameter P5 | $piecewise(2, lt(1, 0.5), 3)$ | constant |
| Initial value of parameter P6 | $piecewise(2, leq(1, 0.5), 3)$ | constant |
| Initial value of parameter P7 | $piecewise(2, eq(P1, pi), 3)$ | constant |
| Initial value of parameter P8 | $piecewise(2, neq(P1, pi), 3)$ | constant |
| Initial value of parameter P9 | $factorial(4)$ | constant |
| Initial value of parameter P10 | $piecewise(2, and(true, false), 3)$ | constant |
| Initial value of parameter P11 | $piecewise(2, or(true, false), 3)$ | constant |
| Initial value of parameter P12 | $piecewise(2, xor(true, false), 3)$ | constant |
| Initial value of parameter P13 | $piecewise(2, not(false), 3)$ | constant |
| Initial value of parameter P14 | $sec(0.5)$ | constant |
| Initial value of parameter P15 | $csc(4.5)$ | constant |
| Initial value of parameter P16 | $cot(0.2)$ | constant |
| Initial value of parameter P17 | $sinh(0.3)$ | constant |
| Initial value of parameter P18 | $cosh(1.7)$ | constant |
| Initial value of parameter P19 | $arcsec(2.3)$ | constant |
| Initial value of parameter P20 | $arccsc(1.1)$ | constant |
| Initial value of parameter P21 | $arccot(-0.1)$ | constant |
| Initial value of parameter P22 | $arcsinh(99)$ | constant |
| Initial value of parameter P23 | $arccosh(1.34)$ | constant |
| Initial value of parameter P24 | $arctanh(-0.7)$ | constant |
| Initial value of parameter P25 | $arcsech(0.42)$ | constant |
| Initial value of parameter P26 | $arccsch(0.01)$ | constant |
| Initial value of parameter P27 | $arccoth(8.2)$ | constant |
| Initial value of parameter P28 | $exponentiale$ | constant |
| Initial value of parameter P29 | $exp(exponentiale)$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

