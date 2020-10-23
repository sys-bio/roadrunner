(*

category:        Test
synopsis:        A trigger with an nary relational element.
componentTags:   CSymbolTime, EventNoDelay, Parameter
testTags:        NonConstantParameter, UncommonMathML
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

In MathML, it is legal to have 3+ arguments to relational elements such as greater than.  This model checks to make sure interpreters notice the third argument, which changes the trigger such that it never fires.

The model contains:
* 1 parameter (x)

There is one event:

[{width:30em,margin: 1em auto}|  *Event*  |  *Trigger*  | *Event Assignments* |
| _E0 | $time > 0.21 && leq(5, 5, 2)$ | $x = 5$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter x | $3$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

