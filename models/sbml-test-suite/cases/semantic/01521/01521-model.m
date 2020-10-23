(*

category:        Test
synopsis:        Two simultaneous events with a priority using a delay.
componentTags:   CSymbolDelay, EventNoDelay, EventPriority, Parameter, RateRule
testTags:        NonConstantParameter
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, two events fire simultaneously, with the priority of one being set using the delay function for another parameter, making its priority lower, and therefore firing second.

The model contains:
* 2 parameters (P1, P2)

There are 2 events:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  | *Event Assignments* |
| E0 | $P1 > 1.5$ | $delay(P1, 1)$ | $P2 = 3$ |
| E1 | $P1 > 1.5$ | $1$ | $P2 = 5$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | P1 | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter P1 | $0$ | variable |
| Initial value of parameter P2 | $0$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
