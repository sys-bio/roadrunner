(*

category:        Test
synopsis:        A replaced species reference in a replaced reaction.
componentTags:   Compartment, Reaction, Species, comp:ModelDefinition, comp:ReplacedElement, comp:Submodel
testTags:        Amount, NonUnityStoichiometry
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

 This is the base model for the previous 9 tests: a replaced species reference inside a replaced reaction.  Neither the reaction ID nor the species reference ID are used anywhere; this is just to make sure simulator software can handle the situation in general.

The 'flattened' version of this hierarchical model contains:
* 1 species (S1)
* 1 compartment (C)
* 1 species reference (J1_sr)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J1: -> 5S1 | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $3$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
