(*

category:        Test
synopsis:        A hierarchical model with a sub-submodel compartment replacement.
componentTags:   Compartment, Species, comp:ModelDefinition, comp:ReplacedElement, comp:SBaseRef, comp:Submodel
testTags:        Concentration, ConstantSpecies, NonUnityCompartment, MultiCompartment, comp:SubmodelOutput
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

 This model tests replacing a compartment in a sub-submodel with a comparment from the parent model, changing the concentration of the species in that submodel.

The 'flattened' version of this hierarchical model contains:
* 4 species (S1, sub1__S1, sub2__S1, sub2__sub1__S1)
* 3 compartments (C, sub1__C, sub2__C)
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $5$ | constant |
| Initial amount of species sub1__S1 | $7$ | constant |
| Initial amount of species sub2__S1 | $9$ | constant |
| Initial amount of species sub2__sub1__S1 | $7$ | constant |
| Initial volume of compartment 'C' | $10$ | constant |
| Initial volume of compartment 'sub1__C' | $1$ | constant |
| Initial volume of compartment 'sub2__C' | $1$ | constant |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

