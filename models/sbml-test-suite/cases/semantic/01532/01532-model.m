(*

category:        Test
synopsis:        Uncommon MathML used in event delays.
componentTags:   EventWithDelay, Parameter, RateRule
testTags:        NonConstantParameter, UncommonMathML
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

The model tests the various unusual mathematical constructs in event delays.

The model contains:
* 57 parameters (trig, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20, P21, P22, P23, P24, P25, P26, P27, P28, P29, P30, P31, P32, P33, P34, P35, P36, P37, P38, P39, P40, P41, P42, P43, P44, P45, P46, P47, P48, P49, P50, P51, P52, P53, P54, P55)

There are 53 events:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Delay*  | *Event Assignments* |
| E0 | $trig > -5.2$ | $1 * (5.3 + exponentiale)$ | $P0 = 2$ |
| E1 | $trig > -5.2$ | $1 * (5.3 + exp(exponentiale))$ | $P1 = 2$ |
| E2 | $trig > -5.2$ | $1 * (5.3 + abs(-1))$ | $P2 = 2$ |
| E3 | $trig > -5.2$ | $1 * (5.3 + abs(1))$ | $P3 = 2$ |
| E4 | $trig > -5.2$ | $1 * (5.3 + acos(-1))$ | $P4 = 2$ |
| E5 | $trig > -5.2$ | $1 * (5.3 + acos(0.5))$ | $P5 = 2$ |
| E6 | $trig > -5.2$ | $1 * (5.3 + asin(1))$ | $P6 = 2$ |
| E7 | $trig > -5.2$ | $1 * (5.3 + asin(-0.5))$ | $P7 = 2$ |
| E8 | $trig > -5.2$ | $1 * (5.3 + atan(2.8))$ | $P8 = 2$ |
| E9 | $trig > -5.2$ | $1 * (5.3 + atan(-7.09))$ | $P9 = 2$ |
| E10 | $trig > -5.2$ | $1 * (5.3 + ceil(0.5))$ | $P10 = 2$ |
| E11 | $trig > -5.2$ | $1 * (5.3 + ceil(3.55))$ | $P11 = 2$ |
| E12 | $trig > -5.2$ | $1 * (5.3 + ceil(-4.6))$ | $P12 = 2$ |
| E13 | $trig > -5.2$ | $1 * (5.3 + cos(9.1))$ | $P13 = 2$ |
| E14 | $trig > -5.2$ | $1 * (5.3 + cos(-0.22))$ | $P14 = 2$ |
| E15 | $trig > -5.2$ | $1 * (5.3 + exp(0))$ | $P15 = 2$ |
| E16 | $trig > -5.2$ | $1 * (5.3 + exp(1))$ | $P16 = 2$ |
| E17 | $trig > -5.2$ | $1 * (5.3 + exp(0.77))$ | $P17 = 2$ |
| E18 | $trig > -5.2$ | $1 * (5.3 + floor(-4.6))$ | $P18 = 2$ |
| E19 | $trig > -5.2$ | $1 * (5.3 + floor(9.1))$ | $P19 = 2$ |
| E20 | $trig > -5.2$ | $1 * (5.3 + ln(0.2))$ | $P20 = 2$ |
| E21 | $trig > -5.2$ | $1 * (5.3 + ln(1))$ | $P21 = 2$ |
| E22 | $trig > -5.2$ | $1 * (5.3 + log10(0.2))$ | $P22 = 2$ |
| E23 | $trig > -5.2$ | $1 * (5.3 + log10(1))$ | $P23 = 2$ |
| E24 | $trig > -5.2$ | $1 * (5.3 + 1^2)$ | $P24 = 2$ |
| E25 | $trig > -5.2$ | $1 * (5.3 + 2^2)$ | $P25 = 2$ |
| E26 | $trig > -5.2$ | $1 * (5.3 + 1.4^5.1)$ | $P26 = 2$ |
| E27 | $trig > -5.2$ | $1 * (5.3 + 4^2)$ | $P27 = 2$ |
| E28 | $trig > -5.2$ | $1 * (5.3 + 3.1^2)$ | $P28 = 2$ |
| E29 | $trig > -5.2$ | $1 * (5.3 + sqrt(4))$ | $P29 = 2$ |
| E30 | $trig > -5.2$ | $1 * (5.3 + sqrt(7.4))$ | $P30 = 2$ |
| E31 | $trig > -5.2$ | $1 * (5.3 + sin(2.1))$ | $P31 = 2$ |
| E32 | $trig > -5.2$ | $1 * (5.3 + sin(0))$ | $P32 = 2$ |
| E33 | $trig > -5.2$ | $1 * (5.3 + sin(-5.9))$ | $P33 = 2$ |
| E34 | $trig > -5.2$ | $1 * (5.3 + tan(0))$ | $P34 = 2$ |
| E35 | $trig > -5.2$ | $1 * (5.3 + tan(1.11))$ | $P35 = 2$ |
| E36 | $trig > -5.2$ | $1 * (5.3 + tan(-6))$ | $P36 = 2$ |
| E37 | $trig > -5.2$ | $1 * (5.3 + sec(0.5))$ | $P37 = 2$ |
| E38 | $trig > -5.2$ | $1 * (5.3 + csc(4.5))$ | $P38 = 2$ |
| E39 | $trig > -5.2$ | $1 * (5.3 + cot(0.2))$ | $P39 = 2$ |
| E40 | $trig > -5.2$ | $1 * (5.3 + sinh(0.3))$ | $P40 = 2$ |
| E41 | $trig > -5.2$ | $1 * (5.3 + cosh(1.7))$ | $P41 = 2$ |
| E42 | $trig > -5.2$ | $1 * (5.3 + arcsec(2.3))$ | $P42 = 2$ |
| E43 | $trig > -5.2$ | $1 * (5.3 + arccsc(1.1))$ | $P43 = 2$ |
| E44 | $trig > -5.2$ | $1 * (5.3 + arccot(-0.1))$ | $P44 = 2$ |
| E45 | $trig > -5.2$ | $1 * (5.3 + arcsinh(99))$ | $P45 = 2$ |
| E46 | $trig > -5.2$ | $1 * (5.3 + arccosh(1.34))$ | $P46 = 2$ |
| E47 | $trig > -5.2$ | $1 * (5.3 + arctanh(-0.7))$ | $P47 = 2$ |
| E48 | $trig > -5.2$ | $1 * (5.3 + arcsech(0.42))$ | $P48 = 2$ |
| E49 | $trig > -5.2$ | $1 * (5.3 + arccsch(0.01))$ | $P49 = 2$ |
| E50 | $trig > -5.2$ | $1 * (5.3 + arccoth(8.2))$ | $P50 = 2$ |
| E51 | $trig > -5.2$ | $1 * (5.3 + times())$ | $P51 = 2$ |
| E52 | $trig > -5.2$ | $1 * plus(5.3)$ | $P52 = 2$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | trig | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter trig | $-5.3$ | variable |
| Initial value of parameter P0 | $0$ | variable |
| Initial value of parameter P1 | $0$ | variable |
| Initial value of parameter P2 | $0$ | variable |
| Initial value of parameter P3 | $0$ | variable |
| Initial value of parameter P4 | $0$ | variable |
| Initial value of parameter P5 | $0$ | variable |
| Initial value of parameter P6 | $0$ | variable |
| Initial value of parameter P7 | $0$ | variable |
| Initial value of parameter P8 | $0$ | variable |
| Initial value of parameter P9 | $0$ | variable |
| Initial value of parameter P10 | $0$ | variable |
| Initial value of parameter P11 | $0$ | variable |
| Initial value of parameter P12 | $0$ | variable |
| Initial value of parameter P13 | $0$ | variable |
| Initial value of parameter P14 | $0$ | variable |
| Initial value of parameter P15 | $0$ | variable |
| Initial value of parameter P16 | $0$ | variable |
| Initial value of parameter P17 | $0$ | variable |
| Initial value of parameter P18 | $0$ | variable |
| Initial value of parameter P19 | $0$ | variable |
| Initial value of parameter P20 | $0$ | variable |
| Initial value of parameter P21 | $0$ | variable |
| Initial value of parameter P22 | $0$ | variable |
| Initial value of parameter P23 | $0$ | variable |
| Initial value of parameter P24 | $0$ | variable |
| Initial value of parameter P25 | $0$ | variable |
| Initial value of parameter P26 | $0$ | variable |
| Initial value of parameter P27 | $0$ | variable |
| Initial value of parameter P28 | $0$ | variable |
| Initial value of parameter P29 | $0$ | variable |
| Initial value of parameter P30 | $0$ | variable |
| Initial value of parameter P31 | $0$ | variable |
| Initial value of parameter P32 | $0$ | variable |
| Initial value of parameter P33 | $0$ | variable |
| Initial value of parameter P34 | $0$ | variable |
| Initial value of parameter P35 | $0$ | variable |
| Initial value of parameter P36 | $0$ | variable |
| Initial value of parameter P37 | $0$ | variable |
| Initial value of parameter P38 | $0$ | variable |
| Initial value of parameter P39 | $0$ | variable |
| Initial value of parameter P40 | $0$ | variable |
| Initial value of parameter P41 | $0$ | variable |
| Initial value of parameter P42 | $0$ | variable |
| Initial value of parameter P43 | $0$ | variable |
| Initial value of parameter P44 | $0$ | variable |
| Initial value of parameter P45 | $0$ | variable |
| Initial value of parameter P46 | $0$ | variable |
| Initial value of parameter P47 | $0$ | variable |
| Initial value of parameter P48 | $0$ | variable |
| Initial value of parameter P49 | $0$ | variable |
| Initial value of parameter P50 | $0$ | variable |
| Initial value of parameter P51 | $0$ | variable |
| Initial value of parameter P52 | $0$ | variable |
| Initial value of parameter P53 | $0$ | variable |
| Initial value of parameter P54 | $0$ | variable |
| Initial value of parameter P55 | $0$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
