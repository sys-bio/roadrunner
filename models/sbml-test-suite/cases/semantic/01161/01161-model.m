(*

category:        Test
synopsis:        A hierarchical model whose submodel has a local parameter.
componentTags:   Compartment, Parameter, Reaction, Species, comp:ModelDefinition, comp:Submodel
testTags:        Amount, HasOnlySubstanceUnits, LocalParameters, comp:SubmodelOutput
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

 A quick test to make sure a submodel local parameter is left alone and not renamed without a commensurate renaming of its reference in the kinetic law as part of any flattening routine a simulator might use, especially when there is a 'shadow' parameter of the same name.

The 'flattened' version of this hierarchical model contains:
* 1 species (sub1__s1)
* 1 parameter (sub1__p1)
* 1 compartment (sub1__C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| -> sub1__s1 | $p1 * sub1__s1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species sub1__s1 | $0.001$ | variable |
| Initial value of parameter sub1__p1 | $100$ | constant |
| Initial volume of compartment 'sub1__C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

