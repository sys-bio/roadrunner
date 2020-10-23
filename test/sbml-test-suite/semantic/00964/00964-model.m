(*

category:      Test
synopsis:      Three competing events with the same priority, jointly causing a parameter to monotonically increase, checking to make sure one event doesn't severely out-compete the other two.   NOTE:  STOCHASTIC TEST. This test is designed to fail only one time in a million, but it still may happen.
componentTags: AssignmentRule, CSymbolTime, EventNoDelay, EventPriority, Parameter
testTags:      EventIsNotPersistent, NonConstantParameter, RandomEventExecution
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Analytic

This model contains three events with the same trigger, the same priority, and all set 'persistent=false', and all of which disable the trigger of the others.  This means that every .01 seconds, one fires and the others do not, at random, and increases the parameters Q, R, and T, respectively.  A fourth parameter, S, is assigned the value of Q+R+T, meaning that it doesn't matter which one fires; S will increase monotonically.  A final parameter, 'error' checks to make sure none of the three parameters are chosen more frequently than the other two--if the difference gets higher than 5, it triggers.  From our simulations, this should happen less often than one time in a million, so if you see it happen in your simulator more often than once, there's a good chance something is wrong.
 

The model contains:
* 7 parameters (S, Q, R, T, reset, tolerance, error)

There are 4 events:

[{width:40em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  |  *Persistent*  | *Event Assignments* |
| Qinc | $geq(time - reset, 0.01)$ | $1$ | false | $reset = time$ |
|  |  |  |  | $Q = Q + 0.01$ |
| Rinc | $geq(time - reset, 0.01)$ | $1$ | false | $reset = time$ |
|  |  |  |  | $R = R + 0.01$ |
| Tinc | $geq(time - reset, 0.01)$ | $1$ | false | $reset = time$ |
|  |  |  |  | $T = T + 0.01$ |
| error_check | $or(geq(abs(Q - R), tolerance), geq(abs(Q - T), tolerance), geq(abs(T - R), tolerance))$ | (unset) | true | $error = 1$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | S | $Q + R + T$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter tolerance | $5$ | constant |
| Initial value of parameter S | $Q + R + T$ | variable |
| Initial value of parameter Q | $0$ | variable |
| Initial value of parameter R | $0$ | variable |
| Initial value of parameter T | $0$ | variable |
| Initial value of parameter reset | $0$ | variable |
| Initial value of parameter error | $0$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

