(*

category:        Test
synopsis:        Use of a Boolean in an event assignment.
componentTags:   CSymbolTime, EventNoDelay, Parameter
testTags:        BoolNumericSwap, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Numeric
packagesPresent: 

This event assigns a value of 'true' to the parameter p1.

The model contains:
* 1 parameter (p1)

There is one event:

[{width:30em,margin: 1em auto}|  *Event*  |  *Trigger*  | *Event Assignments* |
| E0 | $time > 5.5$ | $p1 = true$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p1 | $5$ | variable |]

*)
