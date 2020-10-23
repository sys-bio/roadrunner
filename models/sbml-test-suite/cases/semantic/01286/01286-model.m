(*

category:        Test
synopsis:        Use of a Boolean in an event priority.
componentTags:   CSymbolTime, EventNoDelay, EventPriority, Parameter
testTags:        BoolNumericSwap, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Analytic
packagesPresent: 

The two events have priorities of 'true' and 'false', causing the 'true' priority to fire first.

The model contains:
* 1 parameter (p1)

There are 2 events:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  | *Event Assignments* |
| E0 | $time > 5.5$ | $true$ | $p1 = 10$ |
| E1 | $time > 5.5$ | $false$ | $p1 = 20$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p1 | $5$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
