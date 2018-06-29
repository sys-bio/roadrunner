(*

category:        Test
synopsis:        A hierarchical model with a reaction modified by an extent conversion factor.
componentTags:   Compartment, Parameter, Reaction, Species, comp:ExternalModelDefinition, comp:ReplacedElement, comp:SBaseRef, comp:Submodel
testTags:        Amount, comp:ExtentConversionFactor
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: comp

 The reaction in the submodel must be converted by the extent conversion factor, separated by a level of submodel indirection, and in an external model.  The species and compartment are replaced in the top-level model.

The 'flattened' version of this hierarchical model contains:
* 1 species (s1)
* 1 parameter (extentconv)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| sub2__sub1__J0: -> s1 | $extentconv * 10$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species s1 | $0$ | variable |
| Initial value of parameter extentconv | $1000$ | constant |
| Initial volume of compartment 'C' | $1$ | constant |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

*)
