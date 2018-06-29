(*

category:        Test
synopsis:        An event that fires at t0 that uses a parameter in its own assignment.
componentTags:   EventNoDelay, InitialAssignment, Parameter
testTags:        EventT0Firing, InitialValueReassigned, NonConstantParameter
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, an event that would fire at t0, and use a parameter in its own assignment.  However, though its 'value' attribute would trigger the event, its initial assignment will not, meaning the event should not be triggered.

The model contains:
* 1 parameter (P1)

There is one event:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *initialValue*  | *Event Assignments* |
| E0 | $P1 > 0.9$ | false | $P1 = P1 + 2$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter P1 | $0$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
