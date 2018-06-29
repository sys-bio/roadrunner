(*

category:        Test
synopsis:        Use of the 'max' MathML in assignment rules.
componentTags:   AssignmentRule, CSymbolTime, Parameter
testTags:        InitialValueReassigned, L3v2MathML, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Numeric
packagesPresent: 

Three tests of the 'max' MathML construct in assignment rules.

The model contains:
* 3 parameters (p1, p2, p3)

There are 3 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | p1 | $max(-5, -time)$ |
| Assignment | p2 | $max(time)$ |
| Assignment | p3 | $max(5, time)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p1 | $max(-5, -time)$ | variable |
| Initial value of parameter p2 | $max(time)$ | variable |
| Initial value of parameter p3 | $max(5, time)$ | variable |]

*)
