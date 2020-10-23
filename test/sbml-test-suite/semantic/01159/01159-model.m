(*

category:        Test
synopsis:        This hieararchical model deletes a local parameter
componentTags:   Compartment, Parameter, Reaction, Species, comp:Deletion, comp:ModelDefinition, comp:Submodel
testTags:        Amount, HasOnlySubstanceUnits, comp:SubmodelOutput
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

 In the original submodel, there is a local parameter whose value is used in the kinetic law of a reaction.  When it is imported as a submodel, the local parameter is deleted, causing the kinetic law to instead reference a global parameter with the same ID, but a different value.

The 'flattened' version of this hierarchical model contains:
* 1 species (sub1__s1)
* 1 parameter (sub1__p1)
* 1 compartment (sub1__C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| -> sub1__s1 | $sub1__p1 * sub1__s1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species sub1__s1 | $0.001$ | variable |
| Initial value of parameter sub1__p1 | $100$ | constant |
| Initial volume of compartment 'sub1__C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

