(*

category:        Test
synopsis:        Compartments with different capitalizations.
componentTags:   Compartment
testTags:        MultiCompartment, NonUnityCompartment
testType:        TimeCourse
levels:          1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 A simple model to ensure that the simulator is case-sensitive.

The model contains:
* 3 compartments (comp, Comp, cOmP)
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial volume of compartment 'comp' | $3$ | constant |
| Initial volume of compartment 'Comp' | $4$ | constant |
| Initial volume of compartment 'cOmP' | $5$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
