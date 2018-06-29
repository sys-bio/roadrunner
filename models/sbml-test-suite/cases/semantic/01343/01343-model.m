(*

category:        Test
synopsis:        A test of 'tanh' in different contexts
componentTags:   AssignmentRule, CSymbolTime, InitialAssignment, Parameter
testTags:        InitialValueReassigned, NonConstantParameter, UncommonMathML
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic
packagesPresent: 

 The model tests the 'tanh' MathML function in various contexts.  It was mistakenly omitted from tests 957-959.

The model contains:
* 4 parameters (P1, P2, P3, P4)

There are 3 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | P2 | $tanh(-0.4)$ |
| Assignment | P3 | $tanh(time)$ |
| Assignment | P4 | $tanh(-time)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter P1 | $tanh(0.3)$ | constant |
| Initial value of parameter P2 | $tanh(-0.4)$ | variable |
| Initial value of parameter P3 | $tanh(time)$ | variable |
| Initial value of parameter P4 | $tanh(-time)$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
