(*

category:      Test
synopsis:      Competing events with the same priority, jointly causing a parameter to monotonically increase, checking to make sure one event doesn't severly out-compete the other.  NOTE:  STOCHASTIC TEST. This test is designed to fail only one time in a million, but it still may happen.
componentTags: AssignmentRule, CSymbolTime, EventNoDelay, EventPriority, Parameter
testTags:      EventIsNotPersistent, NonConstantParameter, RandomEventExecution
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Analytic

This model contains two events with the same trigger, the same priority, both set 'persistent=false', and both of which disable the trigger of the other.  This means that every .01 seconds, one fires and the other does not, at random, and increases the parameters Q or R, respectively.  A third parameter, S, is assigned the value of Q+R, meaning that it doesn't matter which one fires; S will increase monotonically.  A final parameter, 'error' checks to make sure neither Q nor R are chosen more frequently than the other--if the difference gets higher than 4, it triggers. In our simulations, this happened less often than one time in a million, so if you see it happen in your simulator more often than once, there's a good chance something is wrong.
 
The model contains:
* 5 parameters (S, Q, R, reset, error)

There are 3 events:

[{width:40em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  |  *Persistent*  | *Event Assignments* |
| Qinc | $geq(time - reset, 0.01)$ | $1$ | false | $reset = time$ |
|  |  |  |  | $Q = Q + 0.01$ |
| Rinc | $geq(time - reset, 0.01)$ | $1$ | false | $reset = time$ |
|  |  |  |  | $R = R + 0.01$ |
| error_check | $geq(abs(Q - R), 4)$ | (unset) | true | $error = 1$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | S | $Q + R$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter S | $Q + R$ | variable |
| Initial value of parameter Q | $0$ | variable |
| Initial value of parameter R | $0$ | variable |
| Initial value of parameter reset | $0$ | variable |
| Initial value of parameter error | $0$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

