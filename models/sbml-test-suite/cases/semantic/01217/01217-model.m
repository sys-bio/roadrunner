(*

category:        Test
synopsis:        Mixed assignment rule and initial assignment dependencies
componentTags:   AssignmentRule, InitialAssignment, Parameter
testTags:        InitialValueReassigned
testType:        TimeCourse
levels:          2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, a chain of dependencies is invoked to determine the parameter's various initial values:  each depends on a successive parameter, whose initial value is set by an initial assignment or assignment rule, alternately.  This ensures that the simulator takes into account the entire system of equations, both initial assignments and assignment rules, when determining initial values.

This model is equivalent to 01218, but with the variables in reverse order, in case a simulator relies on listed order to determine values.

The model contains:
* 6 parameters (a, b, c, d, e, f)

There are 3 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | b | $c + 1$ |
| Assignment | d | $e + 1$ |
| Assignment | f | $4 / 2$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter a | $b + 1$ | constant |
| Initial value of parameter c | $d + 1$ | constant |
| Initial value of parameter e | $f + 1$ | constant |
| Initial value of parameter b | $c + 1$ | variable |
| Initial value of parameter d | $e + 1$ | variable |
| Initial value of parameter f | $4 / 2$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

