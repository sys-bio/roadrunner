(*

category:        Test
synopsis:        A fast reaction with a local parameter.
componentTags:   Compartment, Reaction, Species
testTags:        Amount, FastReaction, LocalParameters
testType:        TimeCourse
levels:          1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1
generatedBy:     Analytic
packagesPresent: 

In this model, a fast reaction has a local parameter that ensures that the reaction proceeds in reverse.  

The model contains:
* 2 species (S1, S2)
* 1 local parameter (J0.k1)
* 1 compartment (default_compartment)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |  *Fast*  |
| J0: S1 -> S2 | $k1 * S1 * S2$ | fast |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $5$ | variable |
| Initial concentration of species S2 | $5$ | variable |
| Initial value of local parameter 'J0.k1' | $-1$ | constant |
| Initial volume of compartment 'default_compartment' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
