(*

category:      Test
synopsis:      A simple reaction with stoichiometry set to a parameter, which in turn is set to be the same as its species.
componentTags: AssignmentRule, Compartment, Parameter, Reaction, Species
testTags:      InitialValueReassigned, Amount, AssignedVariableStoichiometry, NonUnityStoichiometry, NonConstantParameter, HasOnlySubstanceUnits
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Numeric

Note:  This test is the L3 version of model 992.

The stoichiometry of the reaction '-> nX' is set to be equal to p1, a parameter that is set by an assignment rule to be equal to X.

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
| Assignment | Xref | $p1$ |
| Assignment | p1 | $X$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species X | $1$ | variable |
| Initial value of parameter k1 | $1$ | constant |
| Initial value of parameter p1 | $X$ | variable |
| Initial volume of compartment 'default_compartment' | $1$ | constant |]

*)

