(*

category:        Test
synopsis:        A hierarchical model with a replacement of a local parameter.
componentTags:   Compartment, Parameter, Reaction, Species, comp:ModelDefinition, comp:ReplacedElement, comp:Submodel
testTags:        Amount, HasOnlySubstanceUnits, comp:SubmodelOutput
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

A local parameter in the submodel is replaced here by a global parameter.

The 'flattened' version of this hierarchical model contains:
* 1 species (sub1__s1)
* 1 parameter (new_p1)
* 1 compartment (sub1__C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| -> sub1__s1 | $new_p1 * sub1__s1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species sub1__s1 | $0.001$ | variable |
| Initial value of parameter new_p1 | $100$ | constant |
| Initial volume of compartment 'sub1__C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

