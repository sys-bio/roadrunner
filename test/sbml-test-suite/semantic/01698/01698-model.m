(*

category:        Test
synopsis:        An event that fires at t0 that uses a parameter in its own assignment.
componentTags:   EventNoDelay, InitialAssignment, Parameter
testTags:        EventT0Firing, InitialValueReassigned, NonConstantParameter
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, an event that fires at t0 uses a parameter in its own assignment.  Also, its 'value' attribute would not trigger the event, but its initial assignment will.

The model contains:
* 1 parameter (P1)

There is one event:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *initialValue*  | *Event Assignments* |
| E0 | $P1 > 0.9$ | false | $P1 = P1 + 2$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter P1 | $2$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
