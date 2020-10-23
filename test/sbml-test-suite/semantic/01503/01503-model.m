(*

category:        Test
synopsis:        Several parameters with algebraic rules, testing various L1 built-in functions acting on constants.
componentTags:   AlgebraicRule, Parameter
testTags:        InitialValueReassigned, UncommonMathML
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 The model tests the various uncommon mathematical constructs, in algebraic rules.

The model contains:
* 10 parameters (P1, P2, P3, P4, P5, P6, P7, P8, P9, P10)

There are 10 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Algebraic | $0$ | $P1 - plus()$ |
| Algebraic | $0$ | $P2 - times()$ |
| Algebraic | $0$ | $P3 - piecewise(1, and(), 0)$ |
| Algebraic | $0$ | $P4 - piecewise(1, or(), 0)$ |
| Algebraic | $0$ | $P5 - piecewise(1, xor(), 0)$ |
| Algebraic | $0$ | $P6 - plus(3)$ |
| Algebraic | $0$ | $P7 - times(4)$ |
| Algebraic | $0$ | $P8 - piecewise(1, and(true), 0)$ |
| Algebraic | $0$ | $P9 - piecewise(1, or(false), 0)$ |
| Algebraic | $0$ | $P10 - piecewise(1, xor(true), 0)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter P1 | $unknown$ | variable |
| Initial value of parameter P2 | $unknown$ | variable |
| Initial value of parameter P3 | $unknown$ | variable |
| Initial value of parameter P4 | $unknown$ | variable |
| Initial value of parameter P5 | $unknown$ | variable |
| Initial value of parameter P6 | $unknown$ | variable |
| Initial value of parameter P7 | $unknown$ | variable |
| Initial value of parameter P8 | $unknown$ | variable |
| Initial value of parameter P9 | $unknown$ | variable |
| Initial value of parameter P10 | $unknown$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
