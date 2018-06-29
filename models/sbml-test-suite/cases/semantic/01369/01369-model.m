(*

category:        Test
synopsis:        A replaced compartment whose ID is used in a reaction.
componentTags:   Compartment, Reaction, Species, comp:ModelDefinition, comp:ReplacedElement, comp:Submodel
testTags:        Amount, MultiCompartment, NonUnityCompartment, comp:SubmodelOutput
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

 In this model, a replaced submodel compartment ID is used in a reaction.

The 'flattened' version of this hierarchical model contains:
* 1 species (S1)
* 2 compartments (C, J1)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| A__J2: -> S1 | $J1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $3$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |
| Initial volume of compartment 'J1' | $5$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
