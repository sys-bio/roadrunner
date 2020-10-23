(*

category:      Test
synopsis:      Testing 'piecewise' without 'otherwise'.
componentTags: InitialAssignment, Parameter
testTags:      InitialValueReassigned, UncommonMathML
testType:      TimeCourse
levels:        2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

 MathML allows the construction of 'piecwise' functions without 'otherwise'.  When all cases are accounted for, this shouldn't make a difference in the output of the model.

The model contains:
* 2 parameters (a, b)
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter a | $piecewise(1, true)$ | constant |
| Initial value of parameter b | $piecewise(1, gt(a, 0), 2, leq(a, 0))$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

