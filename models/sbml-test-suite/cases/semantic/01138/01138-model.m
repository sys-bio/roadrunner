(*

category:        Test
synopsis:        A hierarchical model with a conversion factor
componentTags:   AssignmentRule, Parameter, comp:ModelDefinition, comp:ReplacedElement, comp:Submodel
testTags:        InitialValueReassigned, NonConstantParameter, comp:ConversionFactor, comp:SubmodelOutput
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

 This model is a simple test of a replacement with a conversion factor that affects the assignment rule of another parameter.

The 'flattened' version of this hierarchical model contains:
* 3 parameters (p8, conv, sub1__prel)

There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | sub1__prel | $p8 / conv / 8$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter conv | $0.1$ | constant |
| Initial value of parameter p8 | $8$ | variable |
| Initial value of parameter sub1__prel | $p8 / conv / 8$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

