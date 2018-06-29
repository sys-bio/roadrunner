(*

category:        Test
synopsis:      Several parameters with assignment rules involving time, testing various L1 built-in functions.
componentTags:   AssignmentRule, CSymbolTime, FunctionDefinition, Parameter
testTags:        InitialValueReassigned, NonConstantParameter, UncommonMathML
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic
packagesPresent: 

 The model tests the various mathematical constructs present in L1, as applied to the time variable, in a function definition.  Ironically, this doesn't work in L1, since time isn't present, but hey.

The model contains:
* 26 parameters (P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P13, P14, P15, P16, P18, P19, P20, P22, P29, P31, P32, P34, P35, P36, P37)

There are 26 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | P1 | $time$ |
| Assignment | P2 | $my_abs(time)$ |
| Assignment | P3 | $my_abs(-time)$ |
| Assignment | P4 | $my_acos(time)$ |
| Assignment | P5 | $my_acos(-time)$ |
| Assignment | P6 | $my_asin(time)$ |
| Assignment | P7 | $my_asin(-time)$ |
| Assignment | P8 | $my_atan(time)$ |
| Assignment | P9 | $my_atan(-time)$ |
| Assignment | P10 | $my_ceil(time)$ |
| Assignment | P11 | $my_ceil(-time)$ |
| Assignment | P13 | $my_cos(time)$ |
| Assignment | P14 | $my_cos(-time)$ |
| Assignment | P15 | $my_exp(time)$ |
| Assignment | P16 | $my_exp(-time)$ |
| Assignment | P18 | $my_floor(time)$ |
| Assignment | P19 | $my_floor(-time)$ |
| Assignment | P20 | $my_ln(time + 1)$ |
| Assignment | P22 | $my_log10(time + 1)$ |
| Assignment | P29 | $my_sqrt(time)$ |
| Assignment | P31 | $my_sin(time)$ |
| Assignment | P32 | $my_sin(-time)$ |
| Assignment | P34 | $my_tan(time)$ |
| Assignment | P35 | $my_tan(-time)$ |
| Assignment | P36 | $my_tanh(time)$ |
| Assignment | P37 | $my_tanh(-time)$ |]

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
 | my_ln | x | $ln(x)$ |
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
| Initial value of parameter P1 | $time$ | variable |
| Initial value of parameter P2 | $my_abs(time)$ | variable |
| Initial value of parameter P3 | $my_abs(-time)$ | variable |
| Initial value of parameter P4 | $my_acos(time)$ | variable |
| Initial value of parameter P5 | $my_acos(-time)$ | variable |
| Initial value of parameter P6 | $my_asin(time)$ | variable |
| Initial value of parameter P7 | $my_asin(-time)$ | variable |
| Initial value of parameter P8 | $my_atan(time)$ | variable |
| Initial value of parameter P9 | $my_atan(-time)$ | variable |
| Initial value of parameter P10 | $my_ceil(time)$ | variable |
| Initial value of parameter P11 | $my_ceil(-time)$ | variable |
| Initial value of parameter P13 | $my_cos(time)$ | variable |
| Initial value of parameter P14 | $my_cos(-time)$ | variable |
| Initial value of parameter P15 | $my_exp(time)$ | variable |
| Initial value of parameter P16 | $my_exp(-time)$ | variable |
| Initial value of parameter P18 | $my_floor(time)$ | variable |
| Initial value of parameter P19 | $my_floor(-time)$ | variable |
| Initial value of parameter P20 | $my_ln(time + 1)$ | variable |
| Initial value of parameter P22 | $my_log10(time + 1)$ | variable |
| Initial value of parameter P29 | $my_sqrt(time)$ | variable |
| Initial value of parameter P31 | $my_sin(time)$ | variable |
| Initial value of parameter P32 | $my_sin(-time)$ | variable |
| Initial value of parameter P34 | $my_tan(time)$ | variable |
| Initial value of parameter P35 | $my_tan(-time)$ | variable |
| Initial value of parameter P36 | $my_tanh(time)$ | variable |
| Initial value of parameter P37 | $my_tanh(-time)$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
