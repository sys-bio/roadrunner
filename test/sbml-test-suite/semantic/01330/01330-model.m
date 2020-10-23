(*

category:        Test
synopsis:        A simultaneous event using trigger-time values.
componentTags:   CSymbolTime, EventNoDelay, EventPriority, Parameter
testTags:        EventUsesTriggerTimeValues, NonConstantParameter
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, an event uses trigger-time values, which makes a difference to the results.

The model contains:
* 2 parameters (x, y)

There are 2 events:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  | *Event Assignments* |
| E1 | $time > 3.5$ | $10$ | $x = 7$ |
| E2 | $time > 3.5$ | $5$ | $y = x$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter x | $3$ | variable |
| Initial value of parameter y | $0$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
