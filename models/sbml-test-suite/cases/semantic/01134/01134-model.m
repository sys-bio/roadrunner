(*

category:        Test
synopsis:        A species replacing one thing and replaced by something else, with ports.
componentTags:   Compartment, Species, comp:ModelDefinition, comp:Port, comp:ReplacedBy, comp:ReplacedElement, comp:SBaseRef, comp:Submodel
testTags:        Concentration, MultiCompartment, NonUnityCompartment, comp:SubmodelOutput
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

 This model is exactly the same as model 01133, but additionally contains ports in the parent model. They are (obviously) unused, so this is simply a test to ensure that their presence doesn't confuse the simulator.

The 'flattened' version of this hierarchical model contains:
* 6 species (sub1__S1, sub2__S1, S1, sub3__S1, sub3__sub1__S1, sub3__sub2__S1)
* 7 compartments (C, sub1__C, sub2__C, sub3__C, sub3__sub1__C, sub3__sub2__C, sub3__sub2__sub1__C)
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species sub1__S1 | $2.5$ | variable |
| Initial amount of species sub2__S1 | $9$ | variable |
| Initial concentration of species S1 | $2.5$ | variable |
| Initial amount of species sub3__S1 | $11$ | variable |
| Initial concentration of species sub3__sub1__S1 | $2.5$ | variable |
| Initial amount of species sub3__sub2__S1 | $9$ | variable |
| Initial volume of compartment 'C' | $10$ | variable |
| Initial volume of compartment 'sub1__C' | $1$ | variable |
| Initial volume of compartment 'sub2__C' | $1$ | variable |
| Initial volume of compartment 'sub3__C' | $1$ | variable |
| Initial volume of compartment 'sub3__sub1__C' | $1$ | variable |
| Initial volume of compartment 'sub3__sub2__C' | $1$ | variable |
| Initial volume of compartment 'sub3__sub2__sub1__C' | $1$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

