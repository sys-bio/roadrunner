(*

category:        Test
synopsis:      Several parameters with assignment rules, testing various L1 built-in functions acting on constants.
componentTags:   AssignmentRule, FunctionDefinition, Parameter
testTags:        InitialValueReassigned, UncommonMathML
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic
packagesPresent: 

 The model tests the various mathematical constructs present in L1, in a function definition.

The model contains:
* 36 parameters (P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20, P21, P22, P23, P24, P25, P26, P27, P28, P29, P30, P31, P32, P33, P34, P35, P36, P37)

There are 36 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | P2 | $my_abs(-1)$ |
| Assignment | P3 | $my_abs(1)$ |
| Assignment | P4 | $my_acos(-1)$ |
| Assignment | P5 | $my_acos(0.5)$ |
| Assignment | P6 | $my_asin(1)$ |
| Assignment | P7 | $my_asin(-0.5)$ |
| Assignment | P8 | $my_atan(2.8)$ |
| Assignment | P9 | $my_atan(-7.09)$ |
| Assignment | P10 | $my_ceil(0.5)$ |
| Assignment | P11 | $my_ceil(3.55)$ |
| Assignment | P12 | $my_ceil(-4.6)$ |
| Assignment | P13 | $my_cos(9.1)$ |
| Assignment | P14 | $my_cos(-0.22)$ |
| Assignment | P15 | $my_exp(0)$ |
| Assignment | P16 | $my_exp(1)$ |
| Assignment | P17 | $my_exp(0.77)$ |
| Assignment | P18 | $my_floor(-4.6)$ |
| Assignment | P19 | $my_floor(9.1)$ |
| Assignment | P20 | $my_log(0.2)$ |
| Assignment | P21 | $my_log(1)$ |
| Assignment | P22 | $my_log10(0.2)$ |
| Assignment | P23 | $my_log10(1)$ |
| Assignment | P24 | $my_pow(1, 2)$ |
| Assignment | P25 | $my_pow(2, 2)$ |
| Assignment | P26 | $my_pow(1.4, 5.1)$ |
| Assignment | P27 | $my_sqr(4)$ |
| Assignment | P28 | $my_sqr(3.1)$ |
| Assignment | P29 | $my_sqrt(4)$ |
| Assignment | P30 | $my_sqrt(7.4)$ |
| Assignment | P31 | $my_sin(2.1)$ |
| Assignment | P32 | $my_sin(0)$ |
| Assignment | P33 | $my_sin(-5.9)$ |
| Assignment | P34 | $my_tan(0)$ |
| Assignment | P35 | $my_tan(1.11)$ |
| Assignment | P36 | $my_tan(-6)$ |
| Assignment | P37 | $my_tanh(-0.4)$ |]

The model contains the following function definitions:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | my_abs | x | $abs(x)$ |
 | my_acos | x | $acos(x)$ |
 | my_asin | x | $asin(x)$ |
 | my_atan | x | $atan(x)$ |
 | my_ceil | x | $ceil(x)$ |
 | my_cos | x | $cos(x)$ |
 | my_exp | x | $exp(x)$ |
 | my_floor | x | $floor(x)$ |
 | my_log | x | $ln(x)$ |
 | my_log10 | x | $log10(x)$ |
 | my_pow | x, y | $x^y$ |
 | my_sqr | x | $x^2$ |
 | my_sqrt | x | $sqrt(x)$ |
 | my_sin | x | $sin(x)$ |
 | my_tan | x | $tan(x)$ |
 | my_tanh | x | $tanh(x)$ |
]
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter P2 | $my_abs(-1)$ | variable |
| Initial value of parameter P3 | $my_abs(1)$ | variable |
| Initial value of parameter P4 | $my_acos(-1)$ | variable |
| Initial value of parameter P5 | $my_acos(0.5)$ | variable |
| Initial value of parameter P6 | $my_asin(1)$ | variable |
| Initial value of parameter P7 | $my_asin(-0.5)$ | variable |
| Initial value of parameter P8 | $my_atan(2.8)$ | variable |
| Initial value of parameter P9 | $my_atan(-7.09)$ | variable |
| Initial value of parameter P10 | $my_ceil(0.5)$ | variable |
| Initial value of parameter P11 | $my_ceil(3.55)$ | variable |
| Initial value of parameter P12 | $my_ceil(-4.6)$ | variable |
| Initial value of parameter P13 | $my_cos(9.1)$ | variable |
| Initial value of parameter P14 | $my_cos(-0.22)$ | variable |
| Initial value of parameter P15 | $my_exp(0)$ | variable |
| Initial value of parameter P16 | $my_exp(1)$ | variable |
| Initial value of parameter P17 | $my_exp(0.77)$ | variable |
| Initial value of parameter P18 | $my_floor(-4.6)$ | variable |
| Initial value of parameter P19 | $my_floor(9.1)$ | variable |
| Initial value of parameter P20 | $my_log(0.2)$ | variable |
| Initial value of parameter P21 | $my_log(1)$ | variable |
| Initial value of parameter P22 | $my_log10(0.2)$ | variable |
| Initial value of parameter P23 | $my_log10(1)$ | variable |
| Initial value of parameter P24 | $my_pow(1, 2)$ | variable |
| Initial value of parameter P25 | $my_pow(2, 2)$ | variable |
| Initial value of parameter P26 | $my_pow(1.4, 5.1)$ | variable |
| Initial value of parameter P27 | $my_sqr(4)$ | variable |
| Initial value of parameter P28 | $my_sqr(3.1)$ | variable |
| Initial value of parameter P29 | $my_sqrt(4)$ | variable |
| Initial value of parameter P30 | $my_sqrt(7.4)$ | variable |
| Initial value of parameter P31 | $my_sin(2.1)$ | variable |
| Initial value of parameter P32 | $my_sin(0)$ | variable |
| Initial value of parameter P33 | $my_sin(-5.9)$ | variable |
| Initial value of parameter P34 | $my_tan(0)$ | variable |
| Initial value of parameter P35 | $my_tan(1.11)$ | variable |
| Initial value of parameter P36 | $my_tan(-6)$ | variable |
| Initial value of parameter P37 | $my_tanh(-0.4)$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
