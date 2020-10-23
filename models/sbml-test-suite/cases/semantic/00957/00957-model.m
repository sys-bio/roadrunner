(*

category:      Test
synopsis:      Several parameters with assignment rules, testing various L2v1 built-in functions acting on constants.
componentTags: AssignmentRule, Parameter
testTags:      NonConstantParameter, InitialValueReassigned, UncommonMathML
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

The model tests the various mathematical constructs added in the L2v1 specification.

The model contains:
* 28 parameters (P1, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20, P21, P22, P23, P24, P25, P26, P27, P28, P29)

There are 28 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | P1 | $pi$ |
| Assignment | P3 | $piecewise(2, gt(1, 0.5), 3)$ |
| Assignment | P4 | $piecewise(2, geq(1, 0.5), 3)$ |
| Assignment | P5 | $piecewise(2, lt(1, 0.5), 3)$ |
| Assignment | P6 | $piecewise(2, leq(1, 0.5), 3)$ |
| Assignment | P7 | $piecewise(2, eq(P1, pi), 3)$ |
| Assignment | P8 | $piecewise(2, neq(P1, pi), 3)$ |
| Assignment | P9 | $factorial(4)$ |
| Assignment | P10 | $piecewise(2, and(true, false), 3)$ |
| Assignment | P11 | $piecewise(2, or(true, false), 3)$ |
| Assignment | P12 | $piecewise(2, xor(true, false), 3)$ |
| Assignment | P13 | $piecewise(2, not(false), 3)$ |
| Assignment | P14 | $sec(0.5)$ |
| Assignment | P15 | $csc(4.5)$ |
| Assignment | P16 | $cot(0.2)$ |
| Assignment | P17 | $sinh(0.3)$ |
| Assignment | P18 | $cosh(1.7)$ |
| Assignment | P19 | $arcsec(2.3)$ |
| Assignment | P20 | $arccsc(1.1)$ |
| Assignment | P21 | $arccot(-0.1)$ |
| Assignment | P22 | $arcsinh(99)$ |
| Assignment | P23 | $arccosh(1.34)$ |
| Assignment | P24 | $arctanh(-0.7)$ |
| Assignment | P25 | $arcsech(0.42)$ |
| Assignment | P26 | $arccsch(0.01)$ |
| Assignment | P27 | $arccoth(8.2)$ |
| Assignment | P28 | $exponentiale$ |
| Assignment | P29 | $exp(exponentiale)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter P1 | $pi$ | variable |
| Initial value of parameter P3 | $piecewise(2, gt(1, 0.5), 3)$ | variable |
| Initial value of parameter P4 | $piecewise(2, geq(1, 0.5), 3)$ | variable |
| Initial value of parameter P5 | $piecewise(2, lt(1, 0.5), 3)$ | variable |
| Initial value of parameter P6 | $piecewise(2, leq(1, 0.5), 3)$ | variable |
| Initial value of parameter P7 | $piecewise(2, eq(P1, pi), 3)$ | variable |
| Initial value of parameter P8 | $piecewise(2, neq(P1, pi), 3)$ | variable |
| Initial value of parameter P9 | $factorial(4)$ | variable |
| Initial value of parameter P10 | $piecewise(2, and(true, false), 3)$ | variable |
| Initial value of parameter P11 | $piecewise(2, or(true, false), 3)$ | variable |
| Initial value of parameter P12 | $piecewise(2, xor(true, false), 3)$ | variable |
| Initial value of parameter P13 | $piecewise(2, not(false), 3)$ | variable |
| Initial value of parameter P14 | $sec(0.5)$ | variable |
| Initial value of parameter P15 | $csc(4.5)$ | variable |
| Initial value of parameter P16 | $cot(0.2)$ | variable |
| Initial value of parameter P17 | $sinh(0.3)$ | variable |
| Initial value of parameter P18 | $cosh(1.7)$ | variable |
| Initial value of parameter P19 | $arcsec(2.3)$ | variable |
| Initial value of parameter P20 | $arccsc(1.1)$ | variable |
| Initial value of parameter P21 | $arccot(-0.1)$ | variable |
| Initial value of parameter P22 | $arcsinh(99)$ | variable |
| Initial value of parameter P23 | $arccosh(1.34)$ | variable |
| Initial value of parameter P24 | $arctanh(-0.7)$ | variable |
| Initial value of parameter P25 | $arcsech(0.42)$ | variable |
| Initial value of parameter P26 | $arccsch(0.01)$ | variable |
| Initial value of parameter P27 | $arccoth(8.2)$ | variable |
| Initial value of parameter P28 | $exponentiale$ | variable |
| Initial value of parameter P29 | $exp(exponentiale)$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

