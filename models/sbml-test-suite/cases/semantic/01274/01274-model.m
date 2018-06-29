(*

category:        Test
synopsis:        Use of the 'implies' MathML in initial assignments.
componentTags:   InitialAssignment, Parameter
testTags:        InitialValueReassigned, L3v2MathML
testType:        TimeCourse
levels:          3.2
generatedBy:     Numeric
packagesPresent: 

 In this model, the 'implies' MathML symbol is used in an initial assignment.  'Implies' means 'if A is true, then B is also true', or '(not A) or B'.  With the automatic translation (in l3v2) of boolean values to numeric values, this means that p1, p3, and p4 are '1' (true), while p2 is '0' (false).

The model contains:
* 4 parameters (p1, p2, p3, p4)
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p1 | $implies(true, true)$ | constant |
| Initial value of parameter p2 | $implies(true, false)$ | constant |
| Initial value of parameter p3 | $implies(false, true)$ | constant |
| Initial value of parameter p4 | $implies(false, false)$ | constant |]

*)
