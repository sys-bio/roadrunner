(*

category:        Test
synopsis:        A delayed event using assignment-time values.
componentTags:   CSymbolTime, EventNoDelay, EventWithDelay, Parameter
testTags:        EventUsesAssignmentTimeValues, NonConstantParameter
testType:        TimeCourse
levels:          2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, an event uses assignment-time values, which makes a difference to the results.

The model contains:
* 2 parameters (x, y)

There are 2 events:

[{width:40em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Use values from:*  |  *Delay*  | *Event Assignments* |
| E1 | $time > 3.5$ | Trigger time | $0$ | $x = 7$ |
| E0 | $time > 2.5$ | Assignment time | $2$ | $y = x$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter x | $1$ | variable |
| Initial value of parameter y | $0$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
