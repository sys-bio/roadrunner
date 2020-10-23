(*

category:      Test
synopsis:      A constant parameter that nonetheless varies in its IntialAssignment, but is constant thereafter, being echoed by a second parameter viewing it through a delay.
componentTags: AssignmentRule, CSymbolDelay, CSymbolTime, InitialAssignment, Parameter
testTags:      NonConstantParameter, UncommonMathML
testType:      TimeCourse
levels:        2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

 The model contains a constant parameter x whose InitialAssignment claims that it has been oscillating (the sine function) up until the start of the simulation, at which point it stops changing.  A second parameter (y) echoes this species with a 0.5 second delay, meaning that it starts by 'seeing' the oscillations before flattening out.

The model contains:
* 2 parameters (y, x)

There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | y | $2 + delay(x, 0.5)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter x | $sin(10 * time)$ | constant |
| Initial value of parameter y | $2 + delay(x, 0.5)$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

