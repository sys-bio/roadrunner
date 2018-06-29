(*

category:        Test
synopsis:        Use of a number in a trigger.
componentTags:   EventNoDelay, Parameter
testTags:        BoolNumericSwap, EventT0Firing, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Numeric
packagesPresent: 

 The trigger for this event is simply '3', which is always 'true' in a Boolean context.  Because the event's initialValue is 'false', this results in the event triggering at t0.

The model contains:
* 1 parameter (p1)

There is one event:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *initialValue*  | *Event Assignments* |
| _E0 | $3$ | false | $p1 = 2$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p1 | $5$ | variable |]

*)
