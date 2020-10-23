(*

category:        Test
synopsis:        A hierarchical model from the specification that uses replacements of deletions
componentTags:   Compartment, Reaction, Species, comp:Deletion, comp:ModelDefinition, comp:Port, comp:ReplacedElement, comp:Submodel
testTags:        Amount
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: comp

 Another model from the specification:  this model meticulously recreates the submodel 'enzyme' in the parent model, essentially documenting how it differs from the other submodel 'simple'.  Because the process removes or deletes everything from both submodels, it is not necessary to understand the 'comp' package to successfully simulate this model.

The 'flattened' version of this hierarchical model contains:
* 4 species (S, D, E, ES)
* 1 compartment (comp)

There are 2 reactions:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| S + E -> ES | $(S + E) / ES$ |
| ES -> E + D | $ES / (E + D)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S | $5$ | variable |
| Initial concentration of species D | $10$ | variable |
| Initial concentration of species E | $7$ | variable |
| Initial concentration of species ES | $0.1$ | variable |
| Initial volume of compartment 'comp' | $1$ | constant |]

*)

