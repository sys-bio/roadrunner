(*

category:        Test
synopsis:        A persistent event triggered by a parameter with an initial assignment.
componentTags:   CSymbolTime, EventNoDelay, EventWithDelay, Parameter
testTags:        EventIsPersistent, NonConstantParameter
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, two events have the same trigger, but different delays.  The one to execute first sets the trigger to 'false', and the one that fires second sets the trigger to 'true' again, causing both the fire repeatedly.  A version of tests 1754 and 1758, but without t0 firing or initial assignments.

The model contains:
* 1 parameter (P1)

There are 3 events:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Delay*  | *Event Assignments* |
| E0 | $P1 > 1$ | $2$ | $P1 = 3$ |
| E1 | $P1 > 1$ | $1$ | $P1 = 0$ |
| E2 | $time > 0.65$ | $0$ | $P1 = 2$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter P1 | $0.5$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
