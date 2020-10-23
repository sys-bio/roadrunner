(*

category:      Test
synopsis:      A very simple reaction whose stoichiometry is referenced in other math.
componentTags: AssignmentRule, Compartment, InitialAssignment, Parameter, RateRule, Reaction, Species
testTags:      Amount, InitialValueReassigned, NonConstantParameter, SpeciesReferenceInMath
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Analytic

The reaction in this model has a speciesReference which doesn't change, but is used in various other equations and rules in the model.

The model contains:
* 1 species (X)
* 4 parameters (Y, Z, Q, k1)
* 1 compartment (default_compartment)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| -> X | $k1$ |]


There are 2 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | Y | $Xref$ |
| Assignment | Z | $Xref$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species X | $0$ | variable |
| Initial value of parameter Q | $Xref$ | constant |
| Initial value of parameter k1 | $1$ | constant |
| Initial value of parameter Y | $Xref$ | variable |
| Initial value of parameter Z | $Xref$ | variable |
| Initial volume of compartment 'default_compartment' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

