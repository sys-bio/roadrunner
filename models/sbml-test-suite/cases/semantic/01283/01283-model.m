(*

category:        Test
synopsis:        Use of numbers in boolean contexts.
componentTags:   InitialAssignment, Parameter
testTags:        BoolNumericSwap, InitialValueReassigned
testType:        TimeCourse
levels:          3.2
generatedBy:     Numeric
packagesPresent: 

Three initial assignments that use numeric valus in Boolean contexts.

The model contains:
* 5 parameters (p1, p2, p3, p4, p5)
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p1 | $piecewise(false, true, true)$ | constant |
| Initial value of parameter p2 | $true$ | constant |
| Initial value of parameter p3 | $false$ | constant |
| Initial value of parameter p4 | $true + 3$ | constant |
| Initial value of parameter p5 | $false / 5 - (2 * true)^2$ | constant |]

*)
