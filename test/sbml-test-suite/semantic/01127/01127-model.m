(*

category:        Test
synopsis:        A hierarchical model with a submodel initial assignment and rate rule.
componentTags:   InitialAssignment, Parameter, RateRule, comp:ModelDefinition, comp:Port, comp:ReplacedElement, comp:Submodel
testTags:        InitialValueReassigned, NonConstantParameter, comp:SubmodelOutput
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

 The parent model contains a parameter that replaces a paramter of the submodel. That submodel has an initial assignment and a rate rule that both affect that replaced parameter, and should continue to affect the replacement parameter.

The 'flattened' version of this hierarchical model contains:
* 2 parameters (param1, submod1__subparam2)

There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | param1 | $8 * submod1__subparam2 + param1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter submod1__subparam2 | $6$ | constant |
| Initial value of parameter param1 | $submod1__subparam2 + 6$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

