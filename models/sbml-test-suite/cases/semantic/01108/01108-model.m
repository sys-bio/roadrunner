(*

category:      Test
synopsis:      A simple reaction with stoichiometry set to a parameter, which in turn is set to be the same as its species via an algebraic rule.
componentTags: AlgebraicRule, AssignmentRule, Compartment, Parameter, Reaction, Species
testTags:      InitialValueReassigned, Amount, AssignedVariableStoichiometry, InitialValueReassigned, NonConstantParameter, NonUnityStoichiometry
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Numeric

Note:  This test is the L3 version of model 993.

The stoichiometry of the reaction '-> nX' is set to be equal to p1, a parameter that is set by an algebraic rule to be equal to X.

The model contains:
* 1 species (X)
* 2 parameters (p1, k1)
* 1 compartment (default_compartment)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| -> Xref X | $k1$ |]
Note:  the following stoichiometries are set separately:  Xref


There are 2 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Algebraic | $0$ | $X - p1$ |
| Assignment | Xref | $p1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species X | $1$ | variable |
| Initial value of parameter k1 | $1$ | constant |
| Initial value of parameter p1 | $unknown$ | variable |
| Initial volume of compartment 'default_compartment' | $1$ | constant |]

*)

