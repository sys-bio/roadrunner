(*

category:        Test
synopsis:        This hierarchical model references an external model definition with a submodel with an external model definition.
componentTags:   Compartment, Reaction, Species, comp:ExternalModelDefinition, comp:Submodel
testTags:        Amount, comp:SubmodelOutput
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: comp

This model tests multiple layers of external model definitions, where the model referenced by an external model definition contains a submodel which is itself defined in a third file via an external model definition in the second file.

The 'flattened' version of this hierarchical model contains:
* 7 species (A__S, A__A__E, A__A__D, A__A__ES, A__B__E, A__B__D, A__B__ES)
* 1 compartment (A__comp)

There are 4 reactions:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| A__S + A__A__E -> A__A__ES | $(A__S + A__A__E) / A__A__ES$ |
| A__A__ES -> A__A__E + A__A__D | $A__A__ES / (A__A__E + A__A__D)$ |
| A__S + A__B__E -> A__B__ES | $(A__S + A__B__E) / A__B__ES$ |
| A__B__ES -> A__B__E + A__B__D | $A__B__ES / (A__B__E + A__B__D)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species A__S | $11$ | variable |
| Initial concentration of species A__A__E | $5$ | variable |
| Initial concentration of species A__A__D | $7$ | variable |
| Initial concentration of species A__A__ES | $0.1$ | variable |
| Initial concentration of species A__B__E | $5$ | variable |
| Initial concentration of species A__B__D | $7$ | variable |
| Initial concentration of species A__B__ES | $0.1$ | variable |
| Initial volume of compartment 'A__comp' | $1$ | constant |]

*)

