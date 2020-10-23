(*

category:        Test
synopsis:        Test of nested 0-child MathML constructs plus, times, and, or, and xor in function definitions.
componentTags:   AssignmentRule, FunctionDefinition, Parameter
testTags:        InitialValueReassigned
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

In the official MathML spec, plus, times, and, or, and xor may all have 0 arguments.  When they do, they are supposed return identity for that construct, namely:
 *plus:  0
 *times: 1
 *and:   true
 *or:    false
 *xor:   false

This tests tests these constructs in function definitions.

The model contains:
* 5 parameters (a, b, c, d, e)

There are 5 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | a | $my_plus(1, my_plus(1, my_plus0()))$ |
| Assignment | b | $my_times(2, my_times(2, my_times0()))$ |
| Assignment | c | $piecewise(1, my_and(my_and(my_and0(), true), true), 0)$ |
| Assignment | d | $piecewise(1, my_or(my_or(my_or0(), true), true), 0)$ |
| Assignment | e | $piecewise(1, my_xor(my_xor(my_xor0(), true), true), 0)$ |]

The model contains the following function definitions:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | my_plus0 |  | $plus()$ |
 | my_times0 |  | $times()$ |
 | my_and0 |  | $and()$ |
 | my_or0 |  | $or()$ |
 | my_xor0 |  | $xor()$ |
 | my_plus | x, y | $x + y$ |
 | my_times | x, y | $x * y$ |
 | my_and | x, y | $x && y$ |
 | my_or | x, y | $x || y$ |
 | my_xor | x, y | $xor(x, y)$ |
]
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter a | $my_plus(1, my_plus(1, my_plus0()))$ | variable |
| Initial value of parameter b | $my_times(2, my_times(2, my_times0()))$ | variable |
| Initial value of parameter c | $piecewise(1, my_and(my_and(my_and0(), true), true), 0)$ | variable |
| Initial value of parameter d | $piecewise(1, my_or(my_or(my_or0(), true), true), 0)$ | variable |
| Initial value of parameter e | $piecewise(1, my_xor(my_xor(my_xor0(), true), true), 0)$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
