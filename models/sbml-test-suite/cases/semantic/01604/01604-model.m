(*

category:        Test
synopsis:        A delayed event assignment with assignment-time values using 'time'.
componentTags:   CSymbolTime, EventWithDelay, Parameter
testTags:        EventUsesAssignmentTimeValues, NonConstantParameter
testType:        TimeCourse
levels:          2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

The model ensures that a delayed event with an assignment of 'time' with assignment-time values works even when the requested time points are not the exact effective assignment time.

The model contains:
* 1 parameter (P1)

There is one event:

[{width:40em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Use values from:*  |  *Delay*  | *Event Assignments* |
| E0 | $time >= 4.5$ | Assignment time | $1$ | $P1 = time$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter P1 | $1$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
