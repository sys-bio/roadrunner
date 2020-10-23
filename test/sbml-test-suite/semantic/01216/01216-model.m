(*

category:        Test
synopsis:        Assignment rules with nary relational elements.
componentTags:   AssignmentRule, Parameter
testTags:        InitialValueReassigned, UncommonMathML
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In MathML, it is legal to have 3+ arguments to relational elements such as greater than.  This model checks to make sure interpreters notice the third and further arguments of all five relational functions, which change the results of various 'piecewise' tests.  The model is constructed so that you can't just take the first and second arguments of the function, nor the first and last arguments, but must check each successive pair of variables.

The model contains:
* 10 parameters (a, b, c, d, e, f, g, h, i, j)

There are 10 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | a | $piecewise(1, gt(2, 1, 2), 3)$ |
| Assignment | b | $piecewise(1, lt(1, 2, 1), 4)$ |
| Assignment | c | $piecewise(1, geq(2, 1, 2), 5)$ |
| Assignment | d | $piecewise(1, leq(1, 2, 1), 6)$ |
| Assignment | e | $piecewise(1, eq(1, 1, 2), 7)$ |
| Assignment | f | $piecewise(1, gt(2, 1, 0, 0, -1), 8)$ |
| Assignment | g | $piecewise(1, lt(1, 2, 3, 3, 4), 9)$ |
| Assignment | h | $piecewise(1, geq(2, 1, 0, 1, -1), 10)$ |
| Assignment | i | $piecewise(1, leq(1, 2, 3, 2, 4), 11)$ |
| Assignment | j | $piecewise(1, eq(1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1), 12)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter a | $piecewise(1, gt(2, 1, 2), 3)$ | variable |
| Initial value of parameter b | $piecewise(1, lt(1, 2, 1), 4)$ | variable |
| Initial value of parameter c | $piecewise(1, geq(2, 1, 2), 5)$ | variable |
| Initial value of parameter d | $piecewise(1, leq(1, 2, 1), 6)$ | variable |
| Initial value of parameter e | $piecewise(1, eq(1, 1, 2), 7)$ | variable |
| Initial value of parameter f | $piecewise(1, gt(2, 1, 0, 0, -1), 8)$ | variable |
| Initial value of parameter g | $piecewise(1, lt(1, 2, 3, 3, 4), 9)$ | variable |
| Initial value of parameter h | $piecewise(1, geq(2, 1, 0, 1, -1), 10)$ | variable |
| Initial value of parameter i | $piecewise(1, leq(1, 2, 3, 2, 4), 11)$ | variable |
| Initial value of parameter j | $piecewise(1, eq(1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1), 12)$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

