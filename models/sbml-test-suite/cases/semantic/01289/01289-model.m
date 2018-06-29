(*

category:        Test
synopsis:        Use of a Boolean in a function definition.
componentTags:   FunctionDefinition, InitialAssignment, Parameter
testTags:        BoolNumericSwap, InitialValueReassigned
testType:        TimeCourse
levels:          3.2
generatedBy:     Analytic
packagesPresent: 

 The function definition returns 'true' if the passed-in argument is greater than 5, and 'false' otherwise.  This function is used to set the initial assignment of two parameters.

The model contains:
* 2 parameters (p1, p2)
The model contains the following function definition:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | gt5 | x | $x > 5$ |
]
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p1 | $gt5(3)$ | constant |
| Initial value of parameter p2 | $gt5(8)$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
