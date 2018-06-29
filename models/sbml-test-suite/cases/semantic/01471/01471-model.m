(*

category:        Test
synopsis:        Replacements in external models
componentTags:   Compartment, Reaction, Species, comp:Deletion, comp:ExternalModelDefinition, comp:ModelDefinition, comp:Port, comp:ReplacedBy, comp:ReplacedElement, comp:Submodel
testTags:        Amount, comp:SubmodelOutput
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: comp

 In this modified example from the specification, an external model is referenced by both a ReplacedBy and a ReplacedElement.

The 'flattened' version of this hierarchical model contains:
* 4 species (S, A__E, D, A__ES)
* 1 compartment (comp)

There are 2 reactions:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| A__J0: S + A__E -> A__ES | $(S + A__E) / A__ES$ |
| A__J1: A__ES -> A__E + D | $A__ES / (A__E + D)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S | $1$ | variable |
| Initial concentration of species A__E | $1$ | variable |
| Initial concentration of species D | $1$ | variable |
| Initial concentration of species A__ES | $1$ | variable |
| Initial volume of compartment 'comp' | $1$ | constant |]

*)
