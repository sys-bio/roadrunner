(*

category:        Test
synopsis:        Species in compartments with different capitalizations.
componentTags:   Compartment, Species
testTags:        Amount, MultiCompartment, NonUnityCompartment
testType:        TimeCourse
levels:          1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 A simple model to ensure that the simulator is case-sensitive.

The model contains:
* 3 species (spec, Spec, sPeC)
* 2 compartments (C, c)
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species spec | $3$ | variable |
| Initial concentration of species Spec | $4$ | variable |
| Initial concentration of species sPeC | $5$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |
| Initial volume of compartment 'c' | $2$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
