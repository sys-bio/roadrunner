(*

category:        Test
synopsis:        Several unusual MathML constructs, wrapped in function definitions.
componentTags:   AssignmentRule, FunctionDefinition, Parameter
testTags:        InitialValueReassigned, NonConstantParameter, UncommonMathML
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

The model tests the various uncommon mathematical constructs wrapped in function definitions.

The model contains:
* 28 parameters (P1, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20, P21, P22, P23, P24, P25, P26, P27, P28, P29)

There are 28 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | P1 | $my_pi()$ |
| Assignment | P3 | $my_piecewise(2, my_gt(1, 0.5), 3)$ |
| Assignment | P4 | $my_piecewise(2, my_geq(1, 0.5), 3)$ |
| Assignment | P5 | $my_piecewise(2, my_lt(1, 0.5), 3)$ |
| Assignment | P6 | $my_piecewise(2, my_leq(1, 0.5), 3)$ |
| Assignment | P7 | $my_piecewise(2, my_eq(P1, pi), 3)$ |
| Assignment | P8 | $my_piecewise(2, my_neq(P1, pi), 3)$ |
| Assignment | P9 | $my_factorial(4)$ |
| Assignment | P10 | $my_piecewise(2, my_and(true, false), 3)$ |
| Assignment | P11 | $my_piecewise(2, my_or(true, false), 3)$ |
| Assignment | P12 | $my_piecewise(2, my_xor(true, false), 3)$ |
| Assignment | P13 | $my_piecewise(2, my_not(false), 3)$ |
| Assignment | P14 | $my_sec(0.5)$ |
| Assignment | P15 | $my_csc(4.5)$ |
| Assignment | P16 | $my_cot(0.2)$ |
| Assignment | P17 | $my_sinh(0.3)$ |
| Assignment | P18 | $my_cosh(1.7)$ |
| Assignment | P19 | $my_arcsec(2.3)$ |
| Assignment | P20 | $my_arccsc(1.1)$ |
| Assignment | P21 | $my_arccot(-0.1)$ |
| Assignment | P22 | $my_arcsinh(99)$ |
| Assignment | P23 | $my_arccosh(1.34)$ |
| Assignment | P24 | $my_arctanh(-0.7)$ |
| Assignment | P25 | $my_arcsech(0.42)$ |
| Assignment | P26 | $my_arccsch(0.01)$ |
| Assignment | P27 | $my_arccoth(8.2)$ |
| Assignment | P28 | $my_exponentiale()$ |
| Assignment | P29 | $my_exp(exponentiale)$ |]

The model contains the following function definitions:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | my_pi |  | $pi$ |
 | my_piecewise | x, y, z | $piecewise(x, y, z)$ |
 | my_gt | x, y | $x > y$ |
 | my_geq | x, y | $x >= y$ |
 | my_lt | x, y | $x < y$ |
 | my_leq | x, y | $x <= y$ |
 | my_eq | x, y | $x == y$ |
 | my_neq | x, y | $x != y$ |
 | my_factorial | x | $factorial(x)$ |
 | my_and | x, y | $x && y$ |
 | my_or | x, y | $x || y$ |
 | my_xor | x, y | $xor(x, y)$ |
 | my_not | x | $!x$ |
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
 | my_exponentiale |  | $exponentiale$ |
 | my_exp | x | $exp(x)$ |
]
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter P1 | $my_pi()$ | variable |
| Initial value of parameter P3 | $my_piecewise(2, my_gt(1, 0.5), 3)$ | variable |
| Initial value of parameter P4 | $my_piecewise(2, my_geq(1, 0.5), 3)$ | variable |
| Initial value of parameter P5 | $my_piecewise(2, my_lt(1, 0.5), 3)$ | variable |
| Initial value of parameter P6 | $my_piecewise(2, my_leq(1, 0.5), 3)$ | variable |
| Initial value of parameter P7 | $my_piecewise(2, my_eq(P1, pi), 3)$ | variable |
| Initial value of parameter P8 | $my_piecewise(2, my_neq(P1, pi), 3)$ | variable |
| Initial value of parameter P9 | $my_factorial(4)$ | variable |
| Initial value of parameter P10 | $my_piecewise(2, my_and(true, false), 3)$ | variable |
| Initial value of parameter P11 | $my_piecewise(2, my_or(true, false), 3)$ | variable |
| Initial value of parameter P12 | $my_piecewise(2, my_xor(true, false), 3)$ | variable |
| Initial value of parameter P13 | $my_piecewise(2, my_not(false), 3)$ | variable |
| Initial value of parameter P14 | $my_sec(0.5)$ | variable |
| Initial value of parameter P15 | $my_csc(4.5)$ | variable |
| Initial value of parameter P16 | $my_cot(0.2)$ | variable |
| Initial value of parameter P17 | $my_sinh(0.3)$ | variable |
| Initial value of parameter P18 | $my_cosh(1.7)$ | variable |
| Initial value of parameter P19 | $my_arcsec(2.3)$ | variable |
| Initial value of parameter P20 | $my_arccsc(1.1)$ | variable |
| Initial value of parameter P21 | $my_arccot(-0.1)$ | variable |
| Initial value of parameter P22 | $my_arcsinh(99)$ | variable |
| Initial value of parameter P23 | $my_arccosh(1.34)$ | variable |
| Initial value of parameter P24 | $my_arctanh(-0.7)$ | variable |
| Initial value of parameter P25 | $my_arcsech(0.42)$ | variable |
| Initial value of parameter P26 | $my_arccsch(0.01)$ | variable |
| Initial value of parameter P27 | $my_arccoth(8.2)$ | variable |
| Initial value of parameter P28 | $my_exponentiale()$ | variable |
| Initial value of parameter P29 | $my_exp(exponentiale)$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
