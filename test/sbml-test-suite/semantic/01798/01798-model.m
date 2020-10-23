(*

category:        Test
synopsis:        Default values for an Event's 'useValuesFromTriggerTime'
componentTags:   EventWithDelay, Parameter, RateRule
testTags:        DefaultValue, EventUsesAssignmentTimeValues, EventUsesTriggerTimeValues, NonConstantParameter
testType:        TimeCourse
levels:          2.4, 2.5
generatedBy:     Analytic
packagesPresent: 

 In SBML L2, the value for an event's 'useValuesFromTriggerTime' attribute defaults to 'true'.  This model tests that this value is correctly assumed, in a model where the first event has no 'useValuesFromTriggerTime' attribute, the second has it set to 'false', and the third has it set to 'true'.  The first should follow the third.

The model contains:
* 3 parameters (P1, P2, P3)

There are 3 events:

[{width:40em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Use values from:*  |  *Delay*  | *Event Assignments* |
| E1 | $P1 >= 1.5$ | Trigger time | $2$ | $P1 = P1$ |
| E2 | $P2 >= 1.5$ | Assignment time | $2$ | $P2 = P2$ |
| E3 | $P3 >= 1.5$ | Trigger time | $2$ | $P3 = P3$ |]


There are 3 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | P1 | $1$ |
| Rate | P2 | $1$ |
| Rate | P3 | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter P1 | $0$ | variable |
| Initial value of parameter P2 | $0$ | variable |
| Initial value of parameter P3 | $0$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
