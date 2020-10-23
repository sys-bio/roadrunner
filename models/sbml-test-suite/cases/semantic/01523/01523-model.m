(*

category:        Test
synopsis:        A delayed event that assigns the value of a delayed parameter
componentTags:   CSymbolDelay, EventWithDelay, Parameter, RateRule
testTags:        DelayInEventAssignment, EventUsesTriggerTimeValues, NonConstantParameter
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, a delayed event assigns a value of a delayed parameter from the trigger time of the event, since the value of its 'userValuesFromTriggerTime' attribute is 'true'.

The model contains:
* 2 parameters (P1, P2)

There is one event:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Delay*  | *Event Assignments* |
| E0 | $P1 > 1.5$ | $2$ | $P2 = delay(P1, 1)$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | P1 | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter P1 | $0$ | variable |
| Initial value of parameter P2 | $0$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
