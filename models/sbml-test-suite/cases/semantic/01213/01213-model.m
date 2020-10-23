(*

category:        Test
synopsis:        A delay with an nary relational element.
componentTags:   CSymbolTime, EventWithDelay, Parameter
testTags:        NonConstantParameter, UncommonMathML
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In MathML, it is legal to have 3+ arguments to relational elements such as less than.  This model checks to make sure interpreters notice the third argument, which changes the delay on the event to be 0.3, and not 0.5.

The model contains:
* 1 parameter (x)

There is one event:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Delay*  | *Event Assignments* |
| E0 | $time > 0.01$ | $piecewise(0.5, leq(1, 2, 1), 0.3)$ | $x = 7$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter x | $3$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

