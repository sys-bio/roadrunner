(*

category:        Test
synopsis:        A reaction with a species in a 0-dimensional compartment.
componentTags:   Compartment, Reaction, Species
testTags:        0D-Compartment, Amount, NonUnityCompartment
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In SBML L3v2, there were special rules for when a compartment had a spatialDimensions of 0.  Those rules were removed for SMBL L3, so a model with a 0D compartment now behaves exactly the same as if the compartment had some other dimensionality.  This model tests a non-unity compartment in a reaction, and gives the species an initial concentration, to ensure the compartment's size is being properly handled.

The model contains:
* 1 species (S1)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J: -> S1 | $1.5$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $1$ | variable |
| Initial volume of compartment 'C' | $2$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
