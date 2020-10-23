(*

category:        Test
synopsis:        Hierarchical model with conversion factor test.
componentTags:   AssignmentRule, Parameter, comp:ModelDefinition, comp:ReplacedElement, comp:Submodel
testTags:        InitialValueReassigned, comp:ConversionFactor
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

This model is a simple test of a replacement with a conversion factor that affects the assignment rule of the replaced parameter.

The 'flattened' version of this hierarchical model contains:
* 2 parameters (p8, conv)

There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | p8 | $80 * conv$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter conv | $0.1$ | constant |
| Initial value of parameter p8 | $80 * conv$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

