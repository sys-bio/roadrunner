(*

category:        Test
synopsis:        A species replacing one thing and replaced by something else.
componentTags:   Compartment, Species, comp:ModelDefinition, comp:ReplacedBy, comp:ReplacedElement, comp:SBaseRef, comp:Submodel
testTags:        Concentration, ConstantSpecies, MultiCompartment, NonUnityCompartment, comp:SubmodelOutput
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

 A three-level deep hierarchical model with a replaced compartment, and a top-level species that replaces a three-deep speces, and is replaced in turn by a two-deep species.

The 'flattened' version of this hierarchical model contains:
* 6 species (sub1__S1, sub2__S1, S1, sub3__S1, sub3__sub1__S1, sub3__sub2__S1)
* 7 compartments (C, sub1__C, sub2__C, sub3__C, sub3__sub1__C, sub3__sub2__C, sub3__sub2__sub1__C)
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species sub1__S1 | $2.5$ | constant |
| Initial amount of species sub2__S1 | $9$ | constant |
| Initial concentration of species S1 | $2.5$ | constant |
| Initial amount of species sub3__S1 | $11$ | constant |
| Initial concentration of species sub3__sub1__S1 | $2.5$ | constant |
| Initial amount of species sub3__sub2__S1 | $9$ | constant |
| Initial volume of compartment 'C' | $10$ | constant |
| Initial volume of compartment 'sub1__C' | $1$ | constant |
| Initial volume of compartment 'sub2__C' | $1$ | constant |
| Initial volume of compartment 'sub3__C' | $1$ | constant |
| Initial volume of compartment 'sub3__sub1__C' | $1$ | constant |
| Initial volume of compartment 'sub3__sub2__C' | $1$ | constant |
| Initial volume of compartment 'sub3__sub2__sub1__C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

