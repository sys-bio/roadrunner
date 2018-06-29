(*

category:        Test
synopsis:        A rate rule with no math to a species reference.
componentTags:   Compartment, RateRule, Reaction, Species
testTags:        Amount, AssignedVariableStoichiometry, NoMathML, NonUnityStoichiometry
testType:        TimeCourse
levels:          3.2
generatedBy:     Analytic
packagesPresent: 

 In this test, a rate rule with no child MathML points to a species reference, which does not end up changing.

The model contains:
* 1 species (S1)
* 1 compartment (C1)
* 1 species reference (S1_sr)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S1_sr S1 | $1$ |]
Note:  the following stoichiometries are set separately:  S1_sr


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | S1_sr | $$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $0$ | variable |
| Initial volume of compartment 'C1' | $1$ | constant |
| Initial value of species reference 'S1_sr' | $2$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
