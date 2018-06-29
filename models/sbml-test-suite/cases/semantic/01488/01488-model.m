(*

category:        Test
synopsis:        Various uncommon MathML constructs, wrapped in function definitions.
componentTags:   AssignmentRule, CSymbolTime, FunctionDefinition, Parameter
testTags:        InitialValueReassigned, NonConstantParameter, UncommonMathML
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 The model tests the various trigonometrical constructs added in L2v1, wrapped in function definitions.  Some functions with weird boundary conditions have been set up with the 'piecewise' function so that the results are never 'inf', or 'nan'.

The model contains:
* 26 parameters (P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20, P21, P22, P23, P24, P25, P26)

There are 26 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | P1 | $my_sec(time)$ |
| Assignment | P2 | $my_sec(-time)$ |
| Assignment | P3 | $my_csc(time + 0.001)$ |
| Assignment | P4 | $my_csc(-time - 0.001)$ |
| Assignment | P5 | $my_cot(time + 0.001)$ |
| Assignment | P6 | $my_cot(-time - 0.001)$ |
| Assignment | P7 | $my_sinh(time)$ |
| Assignment | P8 | $my_sinh(-time)$ |
| Assignment | P9 | $my_cosh(time)$ |
| Assignment | P10 | $my_cosh(-time)$ |
| Assignment | P11 | $my_arcsec(time + 1)$ |
| Assignment | P12 | $my_arcsec(-time - 1)$ |
| Assignment | P13 | $my_arccsc(time + 1)$ |
| Assignment | P14 | $my_arccsc(-time - 1)$ |
| Assignment | P15 | $my_arccot(time)$ |
| Assignment | P16 | $my_arccot(-time)$ |
| Assignment | P17 | $my_arcsinh(time)$ |
| Assignment | P18 | $my_arcsinh(-time)$ |
| Assignment | P19 | $my_arccosh(time + 1)$ |
| Assignment | P20 | $my_piecewise(my_arctanh(time), time < 1, 10)$ |
| Assignment | P21 | $my_piecewise(my_arctanh(-time), time < 1, -10)$ |
| Assignment | P22 | $my_piecewise(0, (time <= 0) || (time >= 1), my_arcsech(time))$ |
| Assignment | P23 | $my_arccsch(time + 1)$ |
| Assignment | P24 | $my_arccsch(-time - 1)$ |
| Assignment | P25 | $my_arccoth(time + 1.001)$ |
| Assignment | P26 | $my_arccoth(-time - 1.001)$ |]

The model contains the following function definitions:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | my_piecewise | x, y, z | $piecewise(x, y, z)$ |
 | my_sec | x | $sec(x)$ |
 | my_csc | x | $csc(x)$ |
 | my_cot | x | $cot(x)$ |
 | my_sinh | x | $sinh(x)$ |
 | my_cosh | x | $cosh(x)$ |
 | my_arcsec | x | $arcsec(x)$ |
 | my_arccsc | x | $arccsc(x)$ |
 | my_arccot | x | $arccot(x)$ |
 | my_arcsinh | x | $arcsinh(x)$ |
 | my_arccosh | x | $arccosh(x)$ |
 | my_arctanh | x | $arctanh(x)$ |
 | my_arcsech | x | $arcsech(x)$ |
 | my_arccsch | x | $arccsch(x)$ |
 | my_arccoth | x | $arccoth(x)$ |
 | my_exponentiale |  | $plus(exponentiale)$ |
 | my_exp | x | $exp(x)$ |
]
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter P1 | $my_sec(time)$ | variable |
| Initial value of parameter P2 | $my_sec(-time)$ | variable |
| Initial value of parameter P3 | $my_csc(time + 0.001)$ | variable |
| Initial value of parameter P4 | $my_csc(-time - 0.001)$ | variable |
| Initial value of parameter P5 | $my_cot(time + 0.001)$ | variable |
| Initial value of parameter P6 | $my_cot(-time - 0.001)$ | variable |
| Initial value of parameter P7 | $my_sinh(time)$ | variable |
| Initial value of parameter P8 | $my_sinh(-time)$ | variable |
| Initial value of parameter P9 | $my_cosh(time)$ | variable |
| Initial value of parameter P10 | $my_cosh(-time)$ | variable |
| Initial value of parameter P11 | $my_arcsec(time + 1)$ | variable |
| Initial value of parameter P12 | $my_arcsec(-time - 1)$ | variable |
| Initial value of parameter P13 | $my_arccsc(time + 1)$ | variable |
| Initial value of parameter P14 | $my_arccsc(-time - 1)$ | variable |
| Initial value of parameter P15 | $my_arccot(time)$ | variable |
| Initial value of parameter P16 | $my_arccot(-time)$ | variable |
| Initial value of parameter P17 | $my_arcsinh(time)$ | variable |
| Initial value of parameter P18 | $my_arcsinh(-time)$ | variable |
| Initial value of parameter P19 | $my_arccosh(time + 1)$ | variable |
| Initial value of parameter P20 | $my_piecewise(my_arctanh(time), time < 1, 10)$ | variable |
| Initial value of parameter P21 | $my_piecewise(my_arctanh(-time), time < 1, -10)$ | variable |
| Initial value of parameter P22 | $my_piecewise(0, (time <= 0) || (time >= 1), my_arcsech(time))$ | variable |
| Initial value of parameter P23 | $my_arccsch(time + 1)$ | variable |
| Initial value of parameter P24 | $my_arccsch(-time - 1)$ | variable |
| Initial value of parameter P25 | $my_arccoth(time + 1.001)$ | variable |
| Initial value of parameter P26 | $my_arccoth(-time - 1.001)$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
