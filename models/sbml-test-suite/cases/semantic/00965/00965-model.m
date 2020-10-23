(*

category:      Test
synopsis:      Competing events with the same priority, jointly causing a parameter to monotonically increase, checking to make sure one event doesn't happen too many times in a row.  NOTE:  STOCHASTIC TEST. This test is designed to fail only one time in a million, but it still may happen.
componentTags: AssignmentRule, CSymbolTime, EventNoDelay, EventPriority, Parameter
testTags:      EventIsNotPersistent, EventIsPersistent, EventUsesTriggerTimeValues, NonConstantParameter, RandomEventExecution
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Analytic

This model (the same basic model as 952) contains two events with the same trigger, the same priority, both set 'persistent=false', and both of which disable the trigger of the other.  This means that every .01 seconds, one fires and the other does not, at random, and increases the parameters Q or R, respectively.  A third parameter, S, is assigned the value of Q+R, meaning that it doesn't matter which one fires; S will increase monotonically.

In this test, we also keep track of how many times in a row either Q or R have been chosen, through the parameters Qrun and Rrun.  Another parameter, maxrun, is set by its own events to be max(maxrun, Qrun, Rrun).  A final event checks to see if maxrun has exceeded 20; if so, it is set to 1.  In our simulations, in 1,000,000 runs, maxrun had an average of 13.25, and never exceeded 15 at all, so exceeding 300 is an occurrence that should happen much less than one time in a million.  While this doesn't mean that your simulator will never have a run where maxrun exceeds 300, it does mean that this is a very rare event, and almost certainly means that something is wrong with your simulator:  it may be that the same random number is being chosen multiple times.

The model contains:
* 8 parameters (S, Q, R, reset, Qrun, Rrun, maxrun, error)

There are 5 events:

[{width:40em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  |  *Persistent*  | *Event Assignments* |
| Qinc | $(time - reset) &geq; 0.01$ | $10$ | false | $reset = time$ |
|  |  |  |  | $Rrun = 0$ |
|  |  |  |  | $Qrun = Qrun + 1$ |
|  |  |  |  | $Q = Q + 0.01$ |
| Rinc | $(time - reset) &geq; 0.01$ | $10$ | false | $reset = time$ |
|  |  |  |  | $Qrun = 0$ |
|  |  |  |  | $Rrun = Rrun + 1$ |
|  |  |  |  | $R = R + 0.01$ |
| maxrun_check1 | $Qrun > maxrun$ | $9$ | true | $maxrun = Qrun$ |
| maxrun_check2 | $Rrun > maxrun$ | $8$ | true | $maxrun = Rrun$ |
| error_check | $maxrun > 300$ | $7$ | true | $error = 1$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | S | $Q + R$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter S | $Q + R$ | variable |
| Initial value of parameter Q | $0$ | variable |
| Initial value of parameter R | $0$ | variable |
| Initial value of parameter reset | $0$ | variable |
| Initial value of parameter Qrun | $0$ | variable |
| Initial value of parameter Rrun | $0$ | variable |
| Initial value of parameter maxrun | $0$ | variable |
| Initial value of parameter error | $0$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
Previous version of this file:  
(* 

category:      Test
componentTags: Parameter, EventNoDelay, EventPriority, AssignmentRule, CSymbolTime
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Analytic

The events are:

[{width:30em,margin: 1em auto}| | *Trigger*   | *Delay* | *Assignments* |
 | Qinc | $time - reset >= 0.01$ | $-$   | $Q = Q + 0.01, Qrun = Qrun + 1, Rrun = 0, reset = time$      |
 | Rinc | $time - reset >= 0.01$ | $-$   | $R = R + 0.01, Rrun = Rrun + 1, Qrun = 0, reset = time$      |
 | maxrun_check1 | $Qrun > maxrun$ | $-$ | $maxrun = Qrun$ |
 | maxrun_check2 | $Rrun > maxrun$ | $-$ | $maxrun = Rrun$ |
 | error_check | $maxrun > 300$ | $-$ | $error = 1$      |]


 The initial conditions are as follows:

[{width:30em,margin: 1em auto}| |*Value*|
|Value of parameter Q          |$0$  |
|Value of parameter R          |$0$  |
|Value of parameter S          |$0$  |
|Value of parameter reset      |$0$  |
|Value of parameter Qrun       |$0$  |
|Value of parameter Rrun       |$0$  |
|Value of parameter maxrun     |$0$  |
|Value of parameter error      |$0$  |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.
*)


*/


*/

