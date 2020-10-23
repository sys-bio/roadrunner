(*

category:      Test
synopsis:      Two events with different 'initialValue's, causing one to fire and one to not fire.
componentTags: EventNoDelay, Parameter, RateRule
testTags:      EventT0Firing, NonConstantParameter
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Analytic

The model contains two events, one that assigns '1' to parameter p2, and one that assigns '1' to parameter p3.  Both have the same trigger, p1>=1, which is true at t0, but is false thereafter due to a rate rule.  The first trigger is set 'initialValue="false"' and the second 'initialValue="true"'.  This causes the first to trigger at t0 and the second to not trigger.
 
The model contains:
* 3 parameters (p1, p2, p3)

There are 2 events:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *initialValue*  | *Event Assignments* |
| E0 | $p1 &geq; 1$ | false | $p2 = 1$ |
| E1 | $p1 &geq; 1$ | true | $p3 = 1$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | p1 | $-1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p1 | $-1$ | variable |
| Initial value of parameter p2 | $0$ | variable |
| Initial value of parameter p3 | $0$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

