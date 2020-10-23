(*

category:        Test
synopsis:        A simple aggregate hieararchical model.
componentTags:   Compartment, Reaction, Species, comp:ModelDefinition, comp:Submodel
testTags:        Amount, MultiCompartment, comp:SubmodelOutput
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: comp

 A submodel with a two-step reaction is imported twice into the parent model, creating parallel sets of species affected by parallel reactions.

The 'flattened' version of this hierarchical model contains:
* 8 species (submod1__S, submod1__E, submod1__D, submod1__ES, submod2__S, submod2__E, submod2__D, submod2__ES)
* 2 compartments (submod1__comp, submod2__comp)

There are 4 reactions:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| submod1__S + submod1__E -> submod1__ES | $(submod1__S + submod1__E) / submod1__ES$ |
| submod1__ES -> submod1__E + submod1__D | $submod1__ES / (submod1__E + submod1__D)$ |
| submod2__S + submod2__E -> submod2__ES | $(submod2__S + submod2__E) / submod2__ES$ |
| submod2__ES -> submod2__E + submod2__D | $submod2__ES / (submod2__E + submod2__D)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species submod1__S | $1$ | variable |
| Initial concentration of species submod1__E | $1$ | variable |
| Initial concentration of species submod1__D | $1$ | variable |
| Initial concentration of species submod1__ES | $1$ | variable |
| Initial concentration of species submod2__S | $1$ | variable |
| Initial concentration of species submod2__E | $1$ | variable |
| Initial concentration of species submod2__D | $1$ | variable |
| Initial concentration of species submod2__ES | $1$ | variable |
| Initial volume of compartment 'submod1__comp' | $1$ | constant |
| Initial volume of compartment 'submod2__comp' | $1$ | constant |]

*)

