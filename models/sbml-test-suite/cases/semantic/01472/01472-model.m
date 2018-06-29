(*

category:        Test
synopsis:        An element in an external model submodel as the target of a replacedBy element.
componentTags:   Compartment, Reaction, Species, comp:ExternalModelDefinition, comp:ReplacedBy, comp:SBaseRef, comp:Submodel
testTags:        Amount, comp:SubmodelOutput
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: comp

 In this model, an element is replaced by an element in a submodel of an external model.

The 'flattened' version of this hierarchical model contains:
* 4 species (S, A__submod1__E, D, A__submod1__ES)
* 1 compartment (comp)

There are 2 reactions:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |  *Fast*  |
| A__submod1__J0: S + A__submod1__E -> A__submod1__ES | $(S + A__submod1__E) / A__submod1__ES$ | fast |
| A__submod1__J1: A__submod1__ES -> A__submod1__E + D | $A__submod1__ES / (A__submod1__E + D)$ | fast |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S | $10$ | variable |
| Initial concentration of species A__submod1__E | $1$ | variable |
| Initial concentration of species D | $15$ | variable |
| Initial concentration of species A__submod1__ES | $1$ | variable |
| Initial volume of compartment 'comp' | $1$ | constant |]

*)
