(*

category:        Test
synopsis:        A replaced species whose ID is used in a rate rule.
componentTags:   Compartment, Parameter, RateRule, Species, comp:ModelDefinition, comp:ReplacedBy, comp:ReplacedElement, comp:Submodel
testTags:        Amount, BoundaryCondition, NonConstantParameter, comp:SubmodelOutput
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

 In this model, a replaced submodel species ID is used in a rate rule.

The 'flattened' version of this hierarchical model contains:
* 1 species (J1)
* 1 parameter (A__x)
* 1 compartment (A_C)

There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | A__x | $J1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species J1 | $5$ | variable |
| Initial value of parameter A__x | $0$ | variable |
| Initial volume of compartment 'A_C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
