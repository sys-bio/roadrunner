(*

category:      Test
synopsis:      A very simple reaction with stoichiometry set by an initialAssignment.
componentTags: Compartment, InitialAssignment, Parameter, Reaction, Species
testTags:      InitialValueReassigned, Amount, AssignedConstantStoichiometry, NonUnityStoichiometry
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Analytic

Note:  This test is the L3 version of model 968.

The speciesReference ('Xref') is set with an initialAssignment of 3 for the L3 model, and with stoichiometryMath in the L2 models.  No 'stoichiometry' value is provided in the speciesReference in the reaction itself.

The model contains:
* 1 species (X)
* 1 parameter (k1)
* 1 compartment (default_compartment)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| -> Xref X | $k1$ |]
Note:  the following stoichiometries are set separately:  Xref

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species X | $0$ | variable |
| Initial value of parameter k1 | $1$ | constant |
| Initial volume of compartment 'default_compartment' | $1$ | constant |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

