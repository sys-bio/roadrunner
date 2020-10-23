(*

category:        Test
synopsis:        A hierarchical model with a reaction modified by an extent conversion factor.
componentTags:   Compartment, Parameter, Reaction, Species, comp:ModelDefinition, comp:Submodel
testTags:        Amount, HasOnlySubstanceUnits, comp:ExtentConversionFactor, comp:SubmodelOutput
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: comp

The reaction in the submodel must be converted by the extent conversion factor.

The 'flattened' version of this hierarchical model contains:
* 1 species (sub1__s1)
* 1 parameter (extentconv)
* 1 compartment (sub1__C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| -> sub1__s1 | $extentconv * 10$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species sub1__s1 | $1$ | variable |
| Initial value of parameter extentconv | $1000$ | constant |
| Initial volume of compartment 'sub1__C' | $0$ | constant |]

*)

