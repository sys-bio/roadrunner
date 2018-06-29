(*

category:        Test
synopsis:        A fast reaction with an assigned stoichiometry
componentTags:   Compartment, InitialAssignment, Reaction, Species
testTags:        Amount, AssignedConstantStoichiometry, FastReaction, InitialValueReassigned, NonUnityStoichiometry
testType:        TimeCourse
levels:          3.1
generatedBy:     Analytic
packagesPresent: 

 In this model, a fast reaction has a stoichiometry assigned by an initial assignment.  

The model contains:
* 2 species (S1, S2)
* 1 compartment (default_compartment)
* 1 species reference (S1_stoichiometry)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |  *Fast*  |
| J0: S1_stoichiometry S1 -> S2 | $1 * S1$ | fast |]
Note:  the following stoichiometries are set separately:  S1_stoichiometry

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $5$ | variable |
| Initial concentration of species S2 | $0$ | variable |
| Initial volume of compartment 'default_compartment' | $1$ | constant |
| Initial value of species reference 'S1_stoichiometry' | $2 + 3$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
