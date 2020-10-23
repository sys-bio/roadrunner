(*

category:      Test
synopsis:      Two oscillators; one that echoes the first by a delay.
componentTags: AssignmentRule, CSymbolDelay, CSymbolTime, Parameter
testTags:      NonConstantParameter, InitialValueReassigned, UncommonMathML
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

The model contains one parameter (x) that oscillates (from the 'sin' function), and a second parameter (y) that echoes the first after a 0.2 second delay.

The model contains:
* 2 parameters (x, y)

There are 2 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | x | $sin(10 * time)$ |
| Assignment | y | $2 + delay(x, 0.2)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter x | $sin(10 * time)$ | variable |
| Initial value of parameter y | $2 + delay(x, 0.2)$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

