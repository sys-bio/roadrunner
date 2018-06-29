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
* 3 parameters (p1, p2, p3)
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p1 | $piecewise(0, 5, 1)$ | constant |
| Initial value of parameter p2 | $true && (0 || 0.0)$ | constant |
| Initial value of parameter p3 | $0 || (1 - 1) || (0^3)$ | constant |]

*)
