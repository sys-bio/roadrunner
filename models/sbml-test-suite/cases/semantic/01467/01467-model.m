(*

category:        Test
synopsis:        Hierarchical model with conversion factor test.
componentTags:   AssignmentRule, Parameter, comp:ModelDefinition, comp:ReplacedElement, comp:SBaseRef, comp:Submodel
testTags:        InitialValueReassigned, comp:ConversionFactor
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

 This model repeats model 01137, testing a replacement with a conversion factor that affects the assignment rule of the replaced parameter, except that in this test, the conversion factor affects a sub-submodel variable, instead of a submodel variable directly.

The 'flattened' version of this hierarchical model contains:
* 2 parameters (conv, p8)

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
