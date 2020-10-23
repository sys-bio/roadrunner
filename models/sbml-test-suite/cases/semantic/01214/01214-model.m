(*

category:        Test
synopsis:        A delay with an nary relational element.
componentTags:   CSymbolTime, EventNoDelay, Parameter
testTags:        NonConstantParameter
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In MathML, it is legal to have 3+ arguments to relational elements such as equals.  This model checks to make sure interpreters notice the third argument, which changes the event assignment to be 7, not 5.

The model contains:
* 1 parameter (x)

There is one event:

[{width:30em,margin: 1em auto}|  *Event*  |  *Trigger*  | *Event Assignments* |
| E0 | $time > 0.01$ | $x = piecewise(5, eq(1, 1, 2), 7)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter x | $3$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

