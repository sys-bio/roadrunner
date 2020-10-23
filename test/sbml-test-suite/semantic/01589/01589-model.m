(*

category:        Test
synopsis:        Competing events with the same priority, jointly causing a parameter to monotonically increase, checking to make sure the two events are not exactly evenly distributed.  NOTE:  STOCHASTIC TEST. Your software may fail periodically; it is only supposed to succeed in the majority of cases.
componentTags:   AlgebraicRule, CSymbolTime, EventNoDelay, EventPriority, Parameter
testTags:        EventIsNotPersistent, InitialValueReassigned, NonConstantParameter, RandomEventExecution
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

This model contains two events with the same trigger, the same effective priority, both set 'persistent=false', and both of which disable the trigger of the other.  This means that every .01 seconds, one fires and the other does not, at random, and increases the parameters Q or R, respectively.  A third parameter, S, is assigned the value of Q+R, meaning that it doesn't matter which one fires; S will increase monotonically.  A final parameter, 'error' checks to make sure neither Q nor R are chosen more frequently than the other--if the difference gets higher than 0.2, it triggers.  Note:  The 'errorLow' and 'errorHigh' parameters are a stochastic test, and may not always remain at '0' for all runs.  If your software fails, try running it again with a new random number seed, and it may succeed.  The values were chosen to be reasonable in the vast majority of cases, but still high enough to reveal problems in software that tends to pick both events either evenly or in an unbalanced manner.  This test is a repeat and synthesis of tests 952 and 962, with the additional wrinkle of 'S' being set with an algebraic rule instead of via an assignment rule.

The model contains:
* 7 parameters (reset, Q, R, maxdiff, errorLow, errorHigh, S)

There are 5 events:

[{width:40em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  |  *Persistent*  | *Event Assignments* |
| Qinc | $(time - reset) >= 0.01$ | $0.01$ | false | $reset = time$ |
|  |  |  |  | $Q = Q + 0.01$ |
| Rinc | $(time - reset) >= 0.01$ | $0.01$ | false | $reset = time$ |
|  |  |  |  | $R = R + 0.01$ |
| maxcheck | $abs(Q - R) > maxdiff$ | (unset) | true | $maxdiff = abs(Q - R)$ |
| error_check | $(time >= 99) && (maxdiff < 0.2)$ | (unset) | true | $errorLow = 1$ |
| error_check2 | $maxdiff >= 4$ | (unset) | true | $errorHigh = 1$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Algebraic | $0$ | $Q + R - S$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter reset | $0$ | variable |
| Initial value of parameter Q | $0$ | variable |
| Initial value of parameter R | $0$ | variable |
| Initial value of parameter maxdiff | $0$ | variable |
| Initial value of parameter errorLow | $0$ | variable |
| Initial value of parameter errorHigh | $0$ | variable |
| Initial value of parameter S | $unknown$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
