(*

category:        Test
synopsis:        An initial assignment with an nary relational element.
componentTags:   FunctionDefinition, InitialAssignment, Parameter
testTags:        InitialValueReassigned
testType:        TimeCourse
levels:          2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In MathML, it is legal to have 3+ arguments to relational elements such as less than.  This model checks to make sure interpreters notice the third argument, which changes the result of the 'piecewise' test.  This tests tests these constructs in function definitions.

The model contains:
* 5 parameters (P1, P2, P3, P4, P5)
The model contains the following function definitions:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | my_lt | x, y, z | $x < y < z$ |
 | my_gt | x, y, z | $x > y > z$ |
 | my_leq | x, y, z | $x <= y <= z$ |
 | my_geq | x, y, z | $x >= y >= z$ |
 | my_eq | x, y, z | $x == y == z$ |
]
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter P1 | $piecewise(1, my_lt(1, 2, 1), 3)$ | constant |
| Initial value of parameter P2 | $piecewise(1, my_gt(2, 1, 2), 3)$ | constant |
| Initial value of parameter P3 | $piecewise(1, my_leq(1, 2, 1), 3)$ | constant |
| Initial value of parameter P4 | $piecewise(1, my_geq(2, 1, 2), 3)$ | constant |
| Initial value of parameter P5 | $piecewise(1, my_eq(1, 1, 2), 3)$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
