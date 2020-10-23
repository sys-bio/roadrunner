(*

category:        Test
synopsis:        A hierarchical model from the specification
componentTags:   Compartment, Reaction, Species, comp:Deletion, comp:ExternalModelDefinition, comp:ModelDefinition, comp:Port, comp:ReplacedElement, comp:Submodel
testTags:        Amount, comp:SubmodelOutput
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: comp

 An example from the 'comp' specification:  two submodels, one simple, and one complex, are synchronized to match the complex version:  'S' converted to 'D' via an intermediate instead of via a single step.

The 'flattened' version of this hierarchical model contains:
* 4 species (S, D, A__E, A__ES)
* 1 compartment (comp)

There are 2 reactions:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S + A__E -> A__ES | $(S + A__E) / A__ES$ |
| A__ES -> A__E + D | $A__ES / (A__E + D)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S | $5$ | variable |
| Initial concentration of species D | $10$ | variable |
| Initial concentration of species A__E | $1$ | variable |
| Initial concentration of species A__ES | $1$ | variable |
| Initial volume of compartment 'comp' | $1$ | constant |]

*)
