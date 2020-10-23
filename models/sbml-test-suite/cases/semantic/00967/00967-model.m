(*

category:      Test
synopsis:      Competing events with alternating priorities, so one fires for 5 seconds, then the other fires for the next 5 seconds.
componentTags: CSymbolTime, EventNoDelay, EventPriority, Parameter
testTags:      NonConstantParameter, EventIsNotPersistent
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Analytic

This model contains two events with the same trigger, both set 'persistent=false', and both of which disable the trigger of the other.  This means that every .01 seconds, one fires and the other does not.  One has the priority 'time' and the other the priority '5.001', meaning that for the first 5 seconds, one fires, and for the next 5 seconds, the other fires.
 

The model contains:
* 3 parameters (reset, Q, R)

There are 2 events:

[{width:40em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  |  *Persistent*  | *Event Assignments* |
| Qinc | $(time - reset) &geq; 0.01$ | $time$ | false | $reset = time$ |
|  |  |  |  | $Q = Q + 0.01$ |
| Rinc | $(time - reset) &geq; 0.01$ | $5.001$ | false | $reset = time$ |
|  |  |  |  | $R = R + 0.01$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter reset | $0$ | variable |
| Initial value of parameter Q | $0$ | variable |
| Initial value of parameter R | $0$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

