(*

category:        Test
synopsis:        Nested function definitions
componentTags:   FunctionDefinition, InitialAssignment, Parameter
testTags:        InitialValueReassigned
testType:        TimeCourse
levels:          2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 A slightly complicated nested function definition, nested three deep.

The model contains:
* 1 parameter (p1)
The model contains the following function definitions:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | addone | x | $x + 1$ |
 | addoneagain | x | $addone(x) + 1$ |
 | addoneagainandagain | x | $addoneagain(x) + addone(x) + 1$ |
]
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p1 | $addoneagainandagain(3)$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
