(*

category:        Test
synopsis:        A delayed event using trigger-time values.
componentTags:   CSymbolTime, EventNoDelay, EventWithDelay, Parameter
testTags:        EventUsesTriggerTimeValues, NonConstantParameter
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, an event uses trigger-time values, which makes a difference to the results.

The model contains:
* 2 parameters (x, y)

There are 2 events:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Delay*  | *Event Assignments* |
| E1 | $time > 3.5$ | $0$ | $x = 7$ |
| E0 | $time > 2.5$ | $2$ | $y = x$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter x | $1$ | variable |
| Initial value of parameter y | $0$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
