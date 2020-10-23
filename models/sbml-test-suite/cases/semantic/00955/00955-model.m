(*

category:        Test
synopsis:      Several parameters with assignment rules involving time, testing various L1 built-in functions.
componentTags:   AssignmentRule, CSymbolTime, Parameter
testTags:        InitialValueReassigned, NonConstantParameter, UncommonMathML
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic
packagesPresent: 

 The model tests the various mathematical constructs present in L1, as applied to the time variable.  Ironically, this doesn't work in L1, since time isn't present, but hey.

The model contains:
* 35 parameters (P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P13, P14, P15, P16, P18, P19, P20, P22, P24, P25, P26, P29, P31, P32, P34, P35, P37, P38, P39, P40, P41, P42, P43, P44)

There are 35 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | P1 | $time$ |
| Assignment | P2 | $abs(time)$ |
| Assignment | P3 | $abs(-time)$ |
| Assignment | P4 | $acos(time)$ |
| Assignment | P5 | $acos(-time)$ |
| Assignment | P6 | $asin(time)$ |
| Assignment | P7 | $asin(-time)$ |
| Assignment | P8 | $atan(time)$ |
| Assignment | P9 | $atan(-time)$ |
| Assignment | P10 | $ceil(time - 0.001)$ |
| Assignment | P11 | $ceil(-time - 0.001)$ |
| Assignment | P13 | $cos(time)$ |
| Assignment | P14 | $cos(-time)$ |
| Assignment | P15 | $exp(time)$ |
| Assignment | P16 | $exp(-time)$ |
| Assignment | P18 | $floor(time + 0.001)$ |
| Assignment | P19 | $floor(-time + 0.001)$ |
| Assignment | P20 | $ln(time + 1)$ |
| Assignment | P22 | $log10(time + 1)$ |
| Assignment | P24 | $time^2$ |
| Assignment | P25 | $2^time$ |
| Assignment | P26 | $time^time$ |
| Assignment | P29 | $sqrt(time)$ |
| Assignment | P31 | $sin(time)$ |
| Assignment | P32 | $sin(-time)$ |
| Assignment | P34 | $tan(time)$ |
| Assignment | P35 | $tan(-time)$ |
| Assignment | P37 | $time + 2$ |
| Assignment | P38 | $time - 2$ |
| Assignment | P39 | $time / 2$ |
| Assignment | P40 | $time * 3$ |
| Assignment | P41 | $2 + time$ |
| Assignment | P42 | $2 - time$ |
| Assignment | P43 | $2 / (time + 1)$ |
| Assignment | P44 | $3 * time$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter P1 | $time$ | variable |
| Initial value of parameter P2 | $abs(time)$ | variable |
| Initial value of parameter P3 | $abs(-time)$ | variable |
| Initial value of parameter P4 | $acos(time)$ | variable |
| Initial value of parameter P5 | $acos(-time)$ | variable |
| Initial value of parameter P6 | $asin(time)$ | variable |
| Initial value of parameter P7 | $asin(-time)$ | variable |
| Initial value of parameter P8 | $atan(time)$ | variable |
| Initial value of parameter P9 | $atan(-time)$ | variable |
| Initial value of parameter P10 | $ceil(time - 0.001)$ | variable |
| Initial value of parameter P11 | $ceil(-time - 0.001)$ | variable |
| Initial value of parameter P13 | $cos(time)$ | variable |
| Initial value of parameter P14 | $cos(-time)$ | variable |
| Initial value of parameter P15 | $exp(time)$ | variable |
| Initial value of parameter P16 | $exp(-time)$ | variable |
| Initial value of parameter P18 | $floor(time + 0.001)$ | variable |
| Initial value of parameter P19 | $floor(-time + 0.001)$ | variable |
| Initial value of parameter P20 | $ln(time + 1)$ | variable |
| Initial value of parameter P22 | $log10(time + 1)$ | variable |
| Initial value of parameter P24 | $time^2$ | variable |
| Initial value of parameter P25 | $2^time$ | variable |
| Initial value of parameter P26 | $time^time$ | variable |
| Initial value of parameter P29 | $sqrt(time)$ | variable |
| Initial value of parameter P31 | $sin(time)$ | variable |
| Initial value of parameter P32 | $sin(-time)$ | variable |
| Initial value of parameter P34 | $tan(time)$ | variable |
| Initial value of parameter P35 | $tan(-time)$ | variable |
| Initial value of parameter P37 | $time + 2$ | variable |
| Initial value of parameter P38 | $time - 2$ | variable |
| Initial value of parameter P39 | $time / 2$ | variable |
| Initial value of parameter P40 | $time * 3$ | variable |
| Initial value of parameter P41 | $2 + time$ | variable |
| Initial value of parameter P42 | $2 - time$ | variable |
| Initial value of parameter P43 | $2 / (time + 1)$ | variable |
| Initial value of parameter P44 | $3 * time$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
