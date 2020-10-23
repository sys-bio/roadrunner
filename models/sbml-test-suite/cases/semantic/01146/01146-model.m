(*

category:        Test
synopsis:        A hierarchical model with a time conversion factor affecting an assignment rule in a sub-submodel.
componentTags:   AssignmentRule, CSymbolTime, Parameter, comp:ModelDefinition, comp:Submodel
testTags:        InitialValueReassigned, NonConstantParameter, comp:SubmodelOutput, comp:TimeConversionFactor
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

 The assignment rule in the sub-submodel has 'time' in it, which must be converted by the time conversion factor of the original submodel.

The 'flattened' version of this hierarchical model contains:
* 2 parameters (timeconv, sub1__sub1__t1)

There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | sub1__sub1__t1 | $time / timeconv + 3$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter timeconv | $60$ | constant |
| Initial value of parameter sub1__sub1__t1 | $time / timeconv + 3$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

