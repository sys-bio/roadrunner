(*

category:        Test
synopsis:        A replaced parameter whose ID is used in a reaction.
componentTags:   Compartment, Parameter, Reaction, Species, comp:ModelDefinition, comp:ReplacedElement, comp:Submodel
testTags:        Amount, comp:SubmodelOutput
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

 In this model, a replaced submodel parameter ID is used in a reaction.

The 'flattened' version of this hierarchical model contains:
* 1 species (S1)
* 1 parameter (J1)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| A__J2: -> S1 | $J1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $3$ | variable |
| Initial value of parameter J1 | $5$ | constant |
| Initial volume of compartment 'C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
