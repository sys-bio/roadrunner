(*

category:      Test
synopsis:      Several parameters with trigonometric assignment rules, testing various L2v1 built-in functions acting on parameters.
componentTags: AssignmentRule, CSymbolTime, Parameter
testTags:      NonConstantParameter, InitialValueReassigned, UncommonMathML
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

 The model tests the various trigonometrical constructs added in L2v1, as assignment rules with time.  Some functions with weird boundary conditions have been set up with the 'piecewise' function so that the results are never 'inf', or 'nan'.

The model contains:
* 26 parameters (P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20, P21, P22, P23, P24, P25, P26)

There are 26 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | P1 | $sec(time)$ |
| Assignment | P2 | $sec(-time)$ |
| Assignment | P3 | $csc(time + 0.001)$ |
| Assignment | P4 | $csc(-time - 0.001)$ |
| Assignment | P5 | $cot(time + 0.001)$ |
| Assignment | P6 | $cot(-time - 0.001)$ |
| Assignment | P7 | $sinh(time)$ |
| Assignment | P8 | $sinh(-time)$ |
| Assignment | P9 | $cosh(time)$ |
| Assignment | P10 | $cosh(-time)$ |
| Assignment | P11 | $arcsec(time + 1)$ |
| Assignment | P12 | $arcsec(-time - 1)$ |
| Assignment | P13 | $arccsc(time + 1)$ |
| Assignment | P14 | $arccsc(-time - 1)$ |
| Assignment | P15 | $arccot(time)$ |
| Assignment | P16 | $arccot(-time)$ |
| Assignment | P17 | $arcsinh(time)$ |
| Assignment | P18 | $arcsinh(-time)$ |
| Assignment | P19 | $arccosh(time + 1)$ |
| Assignment | P20 | $piecewise(arctanh(time), lt(time, 1), 10)$ |
| Assignment | P21 | $piecewise(arctanh(-time), lt(time, 1), -10)$ |
| Assignment | P22 | $piecewise(0, or(leq(time, 0), geq(time, 1)), arcsech(time))$ |
| Assignment | P23 | $arccsch(time + 1)$ |
| Assignment | P24 | $arccsch(-time - 1)$ |
| Assignment | P25 | $arccoth(time + 1.001)$ |
| Assignment | P26 | $arccoth(-time - 1.001)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter P1 | $sec(time)$ | variable |
| Initial value of parameter P2 | $sec(-time)$ | variable |
| Initial value of parameter P3 | $csc(time + 0.001)$ | variable |
| Initial value of parameter P4 | $csc(-time - 0.001)$ | variable |
| Initial value of parameter P5 | $cot(time + 0.001)$ | variable |
| Initial value of parameter P6 | $cot(-time - 0.001)$ | variable |
| Initial value of parameter P7 | $sinh(time)$ | variable |
| Initial value of parameter P8 | $sinh(-time)$ | variable |
| Initial value of parameter P9 | $cosh(time)$ | variable |
| Initial value of parameter P10 | $cosh(-time)$ | variable |
| Initial value of parameter P11 | $arcsec(time + 1)$ | variable |
| Initial value of parameter P12 | $arcsec(-time - 1)$ | variable |
| Initial value of parameter P13 | $arccsc(time + 1)$ | variable |
| Initial value of parameter P14 | $arccsc(-time - 1)$ | variable |
| Initial value of parameter P15 | $arccot(time)$ | variable |
| Initial value of parameter P16 | $arccot(-time)$ | variable |
| Initial value of parameter P17 | $arcsinh(time)$ | variable |
| Initial value of parameter P18 | $arcsinh(-time)$ | variable |
| Initial value of parameter P19 | $arccosh(time + 1)$ | variable |
| Initial value of parameter P20 | $piecewise(arctanh(time), lt(time, 1), 10)$ | variable |
| Initial value of parameter P21 | $piecewise(arctanh(-time), lt(time, 1), -10)$ | variable |
| Initial value of parameter P22 | $piecewise(0, or(leq(time, 0), geq(time, 1)), arcsech(time))$ | variable |
| Initial value of parameter P23 | $arccsch(time + 1)$ | variable |
| Initial value of parameter P24 | $arccsch(-time - 1)$ | variable |
| Initial value of parameter P25 | $arccoth(time + 1.001)$ | variable |
| Initial value of parameter P26 | $arccoth(-time - 1.001)$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

