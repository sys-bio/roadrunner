(*

category:      Test
synopsis:      Competing events with the same priority, jointly causing a parameter to monotonically increase, checking to make sure the two events are not exactly evenly distributed.  NOTE:  STOCHASTIC TEST. Your software may fail periodically; it is only supposed to succeed in the majority of cases.
componentTags: AssignmentRule, CSymbolTime, EventNoDelay, EventPriority, Parameter
testTags:      EventIsNotPersistent, NonConstantParameter, RandomEventExecution
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Analytic

This model contains two events with the same trigger, the same priority, both set 'persistent=false', and both of which disable the trigger of the other.  This means that every .01 seconds, one fires and the other does not, at random, and increases the parameters Q or R, respectively.  A third parameter, S, is assigned the value of Q+R, meaning that it doesn't matter which one fires; S will increase monotonically.  A final parameter, 'error' checks to make sure neither Q nor R are chosen more frequently than the other--if the difference gets higher than 0.2, it triggers. 

 Note:  The 'error' parameter is a stochastic test, and may not always remain at '0' for all runs.  If your software fails, try running it again with a new random number seed, and it may succeed.  The value of '0.2' was chosen to be reasonable in the vast majority of cases, but still high enough to reveal problems in software that tends to pick both events exactly evenly.

The model contains:
* 6 parameters (S, Q, R, reset, maxdiff, error)

There are 4 events:

[{width:40em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  |  *Persistent*  | *Event Assignments* |
| Qinc | $geq(time - reset, 0.01)$ | $1$ | false | $reset = time$ |
|  |  |  |  | $Q = Q + 0.01$ |
| Rinc | $geq(time - reset, 0.01)$ | $1$ | false | $reset = time$ |
|  |  |  |  | $R = R + 0.01$ |
| maxcheck | $gt(abs(Q - R), maxdiff)$ | (unset) | true | $maxdiff = abs(Q - R)$ |
| error_check | $and(geq(time, 99), lt(maxdiff, 0.2))$ | (unset) | true | $error = 1$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | S | $Q + R$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter S | $Q + R$ | variable |
| Initial value of parameter Q | $0$ | variable |
| Initial value of parameter R | $0$ | variable |
| Initial value of parameter reset | $0$ | variable |
| Initial value of parameter maxdiff | $0$ | variable |
| Initial value of parameter error | $0$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

