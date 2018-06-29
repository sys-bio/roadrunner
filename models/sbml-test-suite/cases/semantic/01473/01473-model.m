(*

category:        Test
synopsis:        External models with submodels with replaced elements.
componentTags:   Compartment, Reaction, Species, comp:ExternalModelDefinition, comp:ReplacedElement, comp:SBaseRef, comp:Submodel
testTags:        Amount, comp:SubmodelOutput
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: comp

 In this modification of a model from the specification, an external model has submodels referenced by replacedElements.

The 'flattened' version of this hierarchical model contains:
* 4 species (S, D, A__submod1__E, A__submod1__ES)
* 1 compartment (comp)

There are 2 reactions:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| A__submod1__J0: S + A__submod1__E -> A__submod1__ES | $(S + A__submod1__E) / A__submod1__ES$ |
| A__submod1__J1: A__submod1__ES -> A__submod1__E + D | $A__submod1__ES / (A__submod1__E + D)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S | $5$ | variable |
| Initial concentration of species D | $10$ | variable |
| Initial concentration of species A__submod1__E | $1$ | variable |
| Initial concentration of species A__submod1__ES | $1$ | variable |
| Initial volume of compartment 'comp' | $1$ | constant |]

*)
