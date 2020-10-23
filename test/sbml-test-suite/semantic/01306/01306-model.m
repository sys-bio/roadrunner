(*

category:        Test
synopsis:        An empty reaction's identifier used in an assignment rule.
componentTags:   AssignmentRule, CSymbolTime, Parameter, Reaction
testTags:        InitialValueReassigned, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Analytic
packagesPresent: 

 A reaction with no products nor reactants, but which has a kinetic law, has its identifier used in an assignment rule.  Differs from test 01301 because the kinetic law is non-constant.

The model contains:
* 1 parameter (p1)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> | $time$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | p1 | $J0 + 1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter p1 | $J0 + 1$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
