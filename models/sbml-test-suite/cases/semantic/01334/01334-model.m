(*

category:        Test
synopsis:        Two events that fire at t0 with different priorities.
componentTags:   EventNoDelay, EventPriority, Parameter
testTags:        EventT0Firing, EventUsesAssignmentTimeValues, NonConstantParameter
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, two events both fire at t0, with different priorities.  The first assigns a new value to 'x', and the second assigns the value of 'x' to y, using the *assignment* time value, meaning that afterwards, both x and y end up with the same value.

The model contains:
* 2 parameters (x, y)

There are 2 events:

[{width:45em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  |  *initialValue*  |  *Use values from:*  | *Event Assignments* |
| E1 | $x > 2$ | $10$ | false | Trigger time | $x = 7$ |
| E2 | $x > 2$ | $5$ | false | Assignment time | $y = x$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter x | $3$ | variable |
| Initial value of parameter y | $0$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
