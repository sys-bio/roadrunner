(*

category:      Test
synopsis:      Competing events with the same trigger and delay, but different priorities
componentTags: CSymbolTime, EventPriority, EventWithDelay, Parameter
testTags:      NonConstantParameter
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Numeric

 This test ensures that when two events have the same trigger and same delay, they will still execute in the correct order (by priority).

 There are four models that test basically the same thing twice (for e1 and then e2) to ensure that the order in which the events are discovered within the model will not affect the outcome.

The model contains:
* 2 parameters (e1, e2)

There are 4 events:

[{width:40em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  |  *Delay*  | *Event Assignments* |
| E1 | $time &geq; 3$ | $10$ | $1.5$ | $e1 = 3$ |
| E2 | $time &geq; 3$ | $5$ | $1.5$ | $e1 = 2$ |
| E3 | $time &geq; 3$ | $4$ | $1.5$ | $e2 = 3$ |
| E4 | $time &geq; 3$ | $9$ | $1.5$ | $e2 = 2$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter e1 | $0$ | variable |
| Initial value of parameter e2 | $0$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

