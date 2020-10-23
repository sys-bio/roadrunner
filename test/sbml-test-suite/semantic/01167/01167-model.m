(*

category:        Test
synopsis:        A hierarchical model with two levels of external model definitions.
componentTags:   Compartment, Reaction, Species, comp:ExternalModelDefinition, comp:Submodel
testTags:        Amount, comp:SubmodelOutput
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: comp

 The external model definition in the main model file references another external model definition in a second file, which in turn references an actual model in a third file.

The 'flattened' version of this hierarchical model contains:
* 4 species (A__S, A__E, A__D, A__ES)
* 1 compartment (A__comp)

There are 2 reactions:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| A__S + A__E -> A__ES | $(A__S + A__E) / A__ES$ |
| A__ES -> A__E + A__D | $A__ES / (A__E + A__D)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species A__S | $10$ | variable |
| Initial concentration of species A__E | $5$ | variable |
| Initial concentration of species A__D | $7$ | variable |
| Initial concentration of species A__ES | $0.1$ | variable |
| Initial volume of compartment 'A__comp' | $1$ | constant |]

*)

