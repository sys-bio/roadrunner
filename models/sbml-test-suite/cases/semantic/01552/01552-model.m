(*

category:        Test
synopsis:        A reaction stoichiometry with an assignment rule with no math.
componentTags:   AssignmentRule, Compartment, Reaction, Species
testTags:        Amount, AssignedConstantStoichiometry, InitialValueReassigned, NoMathML, NonUnityStoichiometry
testType:        TimeCourse
levels:          3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, the stoichiometry of an element is 2, with an assignment rule to change it that doesn't have any mathML, leaving the value at 2.

The model contains:
* 2 species (S1, S2)
* 1 compartment (C)
* 1 species reference (S2_stoch)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: S1 -> S2_stoch S2 | $0.1$ |]
Note:  the following stoichiometries are set separately:  S2_stoch


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | S2_stoch | $$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $2$ | variable |
| Initial amount of species S2 | $3$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |
| Initial value of species reference 'S2_stoch' | $2$ | variable |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
