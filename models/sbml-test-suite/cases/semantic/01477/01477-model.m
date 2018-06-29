(*

category:        Test
synopsis:        A hierarchical model with a species modified by a conversion factor.
componentTags:   Compartment, Parameter, Reaction, Species, comp:ExternalModelDefinition, comp:ReplacedElement, comp:SBaseRef, comp:Submodel
testTags:        Amount, comp:ConversionFactor
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: comp

 The reaction in the replaced species must be converted by the conversion factor, separated by a level of submodel indirection, and in an external model.  The species and compartment are replaced in the top-level model.

The 'flattened' version of this hierarchical model contains:
* 1 species (s1)
* 1 parameter (cf)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| sub2__sub1__J0: -> s1 | $10$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species s1 | $2$ | variable |
| Initial value of parameter cf | $10$ | constant |
| Initial volume of compartment 'C' | $1$ | constant |]

*)
