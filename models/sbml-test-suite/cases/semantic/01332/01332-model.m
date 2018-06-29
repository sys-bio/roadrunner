(*

category:        Test
synopsis:        Two events that fire at t0 with different priorities.
componentTags:   EventNoDelay, EventPriority, Parameter
testTags:        EventT0Firing, NonConstantParameter
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, two events both fire at t0, with different priorities.  Since both assign to the same variable, the one with the lower priority sets the value going forward in time.

The model contains:
* 1 parameter (x)

There are 2 events:

[{width:40em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  |  *initialValue*  | *Event Assignments* |
| E1 | $x > 2$ | $10$ | false | $x = 7$ |
| E2 | $x > 2$ | $5$ | false | $x = 5$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter x | $3$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
