(*

category:      Test
synopsis:      A very simple reaction with stoichiometry changed by a rate rule.
componentTags: StoichiometryMath, Compartment, Parameter, RateRule, Reaction, Species
testTags:      Amount, AssignedVariableStoichiometry, DefaultValue, NonConstantParameter, NonUnityStoichiometry
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5
generatedBy:   Analytic

Note:  earlier versions of the test suite contained a 3.1 version of this test.
That model was moved to its own test, because it did not have the 'StoichiometryMath'
component.

Over the course of the simulation, the stoichiometry 'Xref' changes at a rate of 0.01/time, so the effective rate of synthesis increases as well, identically to the same model with k1 changing instead of Xref.  The Level 2 versions of this test create a new parameter 'parameterId_0'.

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
| Rate | Xref | $0.01$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species X | $0$ | variable |
| Initial value of parameter k1 | $1$ | variable |
| Initial volume of compartment 'default_compartment' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
