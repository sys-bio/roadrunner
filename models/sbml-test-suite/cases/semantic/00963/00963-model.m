(*

category:      Test
synopsis:      Competing events with different priorities, meaning one always fires and the other does not.
componentTags: CSymbolTime, EventNoDelay, EventPriority, Parameter
testTags:      EventIsNotPersistent, NonConstantParameter
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Analytic

 This model contains two pairs of events with the same two triggers and different priorities, both set 'persistent=false', and both of which disable the trigger of the other.  This means that every .01 seconds, one fires and the other does not, and the higher-priority event is the one that always wins.  Just to make sure that simulators don't accidentally do what was designed to happen here, the two pairs of events come in the same order alphabetically and within the file, but the priorities are reversed, so if you always pick the first event instead of looking at the priority, you will fail one of the tests.

The model contains:
* 6 parameters (Q, R, reset, Q2, R2, reset2)

There are 4 events:

[{width:40em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  |  *Persistent*  | *Event Assignments* |
| Qinc | $geq(time - reset, 0.01)$ | $1$ | false | $reset = time$ |
|  |  |  |  | $Q = Q + 0.01$ |
| Rinc | $geq(time - reset, 0.01)$ | $-1$ | false | $reset = time$ |
|  |  |  |  | $R = R + 0.01$ |
| Qinc2 | $geq(time - reset2, 0.01)$ | $-1$ | false | $reset2 = time$ |
|  |  |  |  | $Q2 = Q2 + 0.01$ |
| Rinc2 | $geq(time - reset2, 0.01)$ | $1$ | false | $reset2 = time$ |
|  |  |  |  | $R2 = R2 + 0.01$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter Q | $0$ | variable |
| Initial value of parameter R | $0$ | variable |
| Initial value of parameter reset | $0$ | variable |
| Initial value of parameter Q2 | $0$ | variable |
| Initial value of parameter R2 | $0$ | variable |
| Initial value of parameter reset2 | $0$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

