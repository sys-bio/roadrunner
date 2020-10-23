(*

category:      Test
synopsis:      An assignment rule with a continually-changing delay. 
componentTags: AssignmentRule, CSymbolDelay, CSymbolTime, Compartment, EventNoDelay, Parameter, RateRule
testTags:      NonConstantParameter, InitialValueReassigned
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

 This model contains a single parameter x changing at a constant rate, and a second parameter y that follows x with no delay at first, but then switching to a delay of 1 at time 1 through an event.

The model contains:
* 3 parameters (y, x, temp)
* 1 compartment (default_compartment)

There is one event:

[{width:30em,margin: 1em auto}|  *Event*  |  *Trigger*  | *Event Assignments* |
| _E0 | $time &geq; 0.99$ | $temp = 1$ |]


There are 2 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | y | $delay(x, temp)$ |
| Rate | x | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter y | $delay(x, temp)$ | variable |
| Initial value of parameter x | $1$ | variable |
| Initial value of parameter temp | $0$ | variable |
| Initial volume of compartment 'default_compartment' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

