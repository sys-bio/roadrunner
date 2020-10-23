(*

category:      Test
synopsis:      A simple reaction with stoichiometry set to be equal to 'time'.
componentTags: AssignmentRule, CSymbolTime, Compartment, Parameter, Reaction, Species
testTags:      InitialValueReassigned, Amount, AssignedVariableStoichiometry, NonUnityStoichiometry
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Numeric

Note:  This test is the L3 version of model 994.

Variable stoichiometry:  the stoichiometry is literally 'time'.

The model contains:
* 1 species (X)
* 1 parameter (k1)
* 1 compartment (default_compartment)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| -> Xref X | $k1$ |]
Note:  the following stoichiometries are set separately:  Xref


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | Xref | $time$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species X | $1$ | variable |
| Initial value of parameter k1 | $1$ | constant |
| Initial volume of compartment 'default_compartment' | $1$ | constant |]

*)

