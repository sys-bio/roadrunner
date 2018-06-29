(*

category:        Test
synopsis:        An event delay after triggering at t0.
componentTags:   EventWithDelay, Parameter
testTags:        EventT0Firing, NonConstantParameter
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, an event triggers at t0, and executes after a delay.

The model contains:
* 1 parameter (x)

There is one event:

[{width:40em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *initialValue*  |  *Delay*  | *Event Assignments* |
| E1 | $x > 2$ | false | $2.5$ | $x = 7$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter x | $3$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
