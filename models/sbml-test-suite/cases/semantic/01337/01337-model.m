(*

category:        Test
synopsis:        Two events that trigger at t0, one of which fails to cancels the other.
componentTags:   EventNoDelay, EventPriority, Parameter
testTags:        EventIsPersistent, EventT0Firing, NonConstantParameter
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, two events trigger at t0, and the first reverses the trigger of the second, but it's not set persistent=false, allowing it to continue.

This is the best test, because it's test 1337.

The model contains:
* 2 parameters (x, y)

There are 2 events:

[{width:40em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  |  *initialValue*  | *Event Assignments* |
| E1 | $x > 2$ | $10$ | false | $x = 0$ |
| E2 | $x > 2$ | $5$ | false | $y = 7$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter x | $3$ | variable |
| Initial value of parameter y | $6$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
