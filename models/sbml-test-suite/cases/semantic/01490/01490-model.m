(*

category:        Test
synopsis:        Test of 1-child MathML constructs plus, times, and, or, and xor in function definitions.
componentTags:   AssignmentRule, FunctionDefinition, Parameter
testTags:        InitialValueReassigned
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic
packagesPresent: 

In the official MathML spec, plus, times, and, or, and xor may all have 1 arguments.  When they do, they are supposed return that argument.  This tests tests these constructs in function definitions.

The model contains:
* 5 parameters (a, b, c, d, e)

There are 5 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | a | $my_plus(1)$ |
| Assignment | b | $my_times(2)$ |
| Assignment | c | $piecewise(1, my_and(true), 0)$ |
| Assignment | d | $piecewise(1, my_or(true), 0)$ |
| Assignment | e | $piecewise(1, my_xor(true), 0)$ |]

The model contains the following function definitions:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | my_plus | x | $plus(x)$ |
 | my_times | x | $times(x)$ |
 | my_and | x | $and(x)$ |
 | my_or | x | $or(x)$ |
 | my_xor | x | $xor(x)$ |
]
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter a | $my_plus(1)$ | variable |
| Initial value of parameter b | $my_times(2)$ | variable |
| Initial value of parameter c | $piecewise(1, my_and(true), 0)$ | variable |
| Initial value of parameter d | $piecewise(1, my_or(true), 0)$ | variable |
| Initial value of parameter e | $piecewise(1, my_xor(true), 0)$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
