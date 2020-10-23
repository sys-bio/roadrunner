(*

category:      Test
synopsis:      Two event where the two event assignments should not interfere with each other. 
componentTags: CSymbolTime, EventWithDelay, Parameter
testTags:      EventUsesAssignmentTimeValues, NonConstantParameter
testType:      TimeCourse
levels:        2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

 This model contains two events that trigger when time > 0.98 and 0.99.  In each, there is one event assignment that changes the value of a parameter, and a second event assignment that uses that parameter to change the value of a second parameter.  However, the second event assignment should not use the changed value from the first parameter, but rather the value it had at the beginning of event assignment.  (This is true regardless of the value of 'useValuesFromTriggerTime', but the value of that flag here happens to be false.  In order to test this in l2v4 and l2v5, there had to be a delay.)

The model contains:
* 4 parameters (x, y, p, q)

There are 2 events:

[{width:40em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Use values from:*  |  *Delay*  | *Event Assignments* |
| _E0 | $time &geq; 0.98$ | Assignment time | $0.1$ | $y = y + x$ |
|  |  |  |  | $x = 2$ |
| _E1 | $time &geq; 0.99$ | Assignment time | $0.1$ | $p = 3$ |
|  |  |  |  | $q = p + 1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter x | $0$ | variable |
| Initial value of parameter y | $0$ | variable |
| Initial value of parameter p | $0$ | variable |
| Initial value of parameter q | $0$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

