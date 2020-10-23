(*

category:        Test
synopsis:        A hierarchical model with a time conversion factor.
componentTags:   AssignmentRule, CSymbolTime, Parameter, comp:ModelDefinition, comp:ReplacedElement, comp:Submodel
testTags:        InitialValueReassigned, NonConstantParameter, comp:TimeConversionFactor
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

This model contains a submodel with an initial assignment using the 'time' csymbol, contained in a model with a time conversion factor.

The 'flattened' version of this hierarchical model contains:
* 2 parameters (timeconv, t2)

There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | t2 | $time / timeconv + 3$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter timeconv | $60$ | constant |
| Initial value of parameter t2 | $time / timeconv + 3$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

