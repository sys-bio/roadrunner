(*

category:      Test
synopsis:      Competing events without priorities, jointly causing a parameter to monotonically increase.
componentTags: AssignmentRule, CSymbolTime, EventNoDelay, Parameter
testTags:      NonConstantParameter, EventIsNotPersistent, InitialValueReassigned
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Analytic

This model contains two events with the same trigger, no priority, both set 'persistent=false', and both of which disable the trigger of the other.  This means that every .01 seconds, one fires and the other does not, at random, and increases the parameters Q or R, respectively.  A third parameter, S, is assigned the value of Q+R, meaning that it doesn't matter which one fires; S will increase monotonically.  This model is identical to 952, except that there are no priorities, so Qinc and Rinc do not have to be chosen between at random, but can be chosen in any ratio.

The model contains:
* 5 parameters (S, Q, R, reset, error)

There are 2 events:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Persistent*  | *Event Assignments* |
| Qinc | $geq(time - reset, 0.01)$ | false | $reset = time$ |
|  |  |  | $Q = Q + 0.01$ |
| Rinc | $geq(time - reset, 0.01)$ | false | $reset = time$ |
|  |  |  | $R = R + 0.01$ |]


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

