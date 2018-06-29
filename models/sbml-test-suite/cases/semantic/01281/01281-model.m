(*

category:        Test
synopsis:        Use of the 'min' MathML in assignment rules.
componentTags:   AssignmentRule, CSymbolTime, Parameter
testTags:        InitialValueReassigned, L3v2MathML, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Numeric
packagesPresent: 

Three tests of the 'min' MathML construct in assignment rules.

The model contains:
* 3 parameters (p1, p2, p3)

There are 3 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | p1 | $min(-5, -time)$ |
| Assignment | p2 | $min(time)$ |
| Assignment | p3 | $min(5, time, 7)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p1 | $min(-5, time)$ | variable |
| Initial value of parameter p2 | $min(time)$ | variable |
| Initial value of parameter p3 | $min(5, time, 7)$ | variable |]

*)
