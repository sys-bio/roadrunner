(*

category:        Test
synopsis:        A hierarchical model with a conversion factor on a compartment.
componentTags:   Compartment, Parameter, RateRule, comp:ModelDefinition, comp:ReplacedElement, comp:Submodel
testTags:        NonConstantCompartment, NonUnityCompartment, comp:ConversionFactor
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

 This model is a simple test of a replacement with a conversion factor that affects the rate rule of the replaced compartment.

The 'flattened' version of this hierarchical model contains:
* 1 parameter (conv)
* 1 compartment (C8)

There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | C8 | $(4 * (C8 / conv) + 3) * conv$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter conv | $0.1$ | constant |
| Initial volume of compartment 'C8' | $8$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

