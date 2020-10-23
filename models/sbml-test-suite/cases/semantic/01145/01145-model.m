(*

category:        Test
synopsis:        A hierarchical model with an extent conversion factor affecting a sub-submodel.
componentTags:   Compartment, Parameter, Reaction, Species, comp:ModelDefinition, comp:Submodel
testTags:        Amount, HasOnlySubstanceUnits, comp:ExtentConversionFactor, comp:SubmodelOutput
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

 The extent conversion factor in the parent model must affect the reaction in the sub-submodel.

The 'flattened' version of this hierarchical model contains:
* 1 species (sub1__sub1__s1)
* 1 parameter (extentconv)
* 1 compartment (sub1__sub1__C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| -> sub1__sub1__s1 | $extentconv * sub1__sub1__s1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species sub1__sub1__s1 | $0.01$ | variable |
| Initial value of parameter extentconv | $10$ | constant |
| Initial volume of compartment 'sub1__sub1__C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

