(*

category:      Test
synopsis:      A very simple reaction with stoichiometry set by an assignmentRule.
componentTags: AssignmentRule, Compartment, Parameter, Reaction, Species
testTags:      Amount, AssignedConstantStoichiometry, NonUnityStoichiometry, InitialValueReassigned
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Analytic

The speciesReference ('Xref') is set with an assignmentRule of 3.  A 'stoichiometry' value of '1' is provided in the speciesReference in the reaction itself, which should be overridden by the assignment rule.

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
| Assignment | Xref | $3$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species X | $0$ | variable |
| Initial value of parameter k1 | $1$ | constant |
| Initial volume of compartment 'default_compartment' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

