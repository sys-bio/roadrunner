(*

category:        Test
synopsis:        An assignment-time event with a function definition.
componentTags:   EventWithDelay, FunctionDefinition, Parameter, RateRule
testTags:        EventUsesAssignmentTimeValues, NonConstantParameter
testType:        TimeCourse
levels:          2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, a delayed event uses the assignment-time value of its parameters in a function definition.

The model contains:
* 1 parameter (P1)

There is one event:

[{width:40em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Use values from:*  |  *Delay*  | *Event Assignments* |
| E0 | $P1 <= 8.9$ | Assignment time | $2$ | $P1 = addEm(P1, 3)$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | P1 | $-1$ |]

The model contains the following function definition:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | addEm | x, y | $x + y$ |
]
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter P1 | $10$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
