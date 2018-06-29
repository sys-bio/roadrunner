(*

category:        Test
synopsis:        Use of the 'rem' MathML in an assignment rule.
componentTags:   AssignmentRule, CSymbolTime, FunctionDefinition, Parameter
testTags:        InitialValueReassigned, L3v2MathML, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Numeric
packagesPresent: 

In this model, the 'rem' MathML symbol is used in an assignment rule function definition.

The model contains:
* 1 parameter (p1)

There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | p1 | $my_rem(time, 2)$ |]

The model contains the following function definition:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | my_rem | x, y | $rem(x, y)$ |
]
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p1 | $my_rem(time, 2)$ | variable |]

*)
