(*

category:        Test
synopsis:        Priorities with nary relational elements.
componentTags:   CSymbolTime, EventNoDelay, EventPriority, Parameter
testTags:        NonConstantParameter, UncommonMathML
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In MathML, it is legal to have 3+ arguments to relational elements such as greater than.  This model checks to make sure interpreters notice the third argument, which changes the priorities on the competing events such that E0 should have the higher priority, firing first, and E1 has the lower priority, firing second, and assigning the value of '7' to x moving forward.

The model contains:
* 1 parameter (x)

There are 2 events:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  | *Event Assignments* |
| E0 | $time > 0.01$ | $piecewise(1, leq(1, 2, 1), 10)$ | $x = 5$ |
| E1 | $time > 0.01$ | $piecewise(10, leq(1, 2, 1), 1)$ | $x = 7$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter x | $3$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

