(*

category:        Test
synopsis:        A persistent event triggered by a parameter with an initial assignment.
componentTags:   EventWithDelay, InitialAssignment, Parameter
testTags:        EventIsPersistent, EventT0Firing, InitialValueReassigned, NonConstantParameter
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, two events fire at t0 based on a parameter set with an initial assignment.  One would cancel the other, were it not persistent.

The model contains:
* 1 parameter (P1)

There are 2 events:

[{width:40em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *initialValue*  |  *Delay*  | *Event Assignments* |
| E0 | $P1 > 1$ | false | $2.3$ | $P1 = 3$ |
| E1 | $P1 > 1$ | false | $1.5$ | $P1 = 0$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter P1 | $3 / 2$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
