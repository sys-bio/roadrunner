(*

category:        Test
synopsis:        A function definition in an event that fires at t0.
componentTags:   EventNoDelay, FunctionDefinition, Parameter
testTags:        EventT0Firing, NonConstantParameter
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, the trigger of an event that fires at t0 uses a function defintion.

The model contains:
* 1 parameter (P1)

There is one event:

[{width:35em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *initialValue*  | *Event Assignments* |
| E0 | $comparePointNine(P1)$ | false | $P1 = 3$ |]

The model contains the following function definition:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | comparePointNine | x | $x > 0.9$ |
]
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter P1 | $1$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
