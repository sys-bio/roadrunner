(*

category:        Test
synopsis:        A hierarchical model with a conversion factor on a species.
componentTags:   Compartment, Parameter, RateRule, Species, comp:ModelDefinition, comp:ReplacedElement, comp:Submodel
testTags:        Amount, HasOnlySubstanceUnits, MultiCompartment, comp:ConversionFactor
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

This model is a simple test of a replacement with a conversion factor that affects the rate rule of the replaced species.

The 'flattened' version of this hierarchical model contains:
* 1 species (s8)
* 1 parameter (conv)
* 2 compartments (C, sub1__C)

There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | s8 | $(4 * (s8 / conv) + 3) * conv$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species s8 | $8$ | variable |
| Initial value of parameter conv | $0.1$ | constant |
| Initial volume of compartment 'C' | $1$ | constant |
| Initial volume of compartment 'sub1__C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

