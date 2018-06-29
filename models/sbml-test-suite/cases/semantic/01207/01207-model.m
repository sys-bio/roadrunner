(*

category:        Test
synopsis:        Variable compartment with explicitly fixed species
componentTags:   AssignmentRule, Compartment, Parameter, RateRule, Species
testTags:        Amount, BoundaryCondition, HasOnlySubstanceUnits, InitialValueReassigned, NonConstantCompartment, NonConstantParameter, NonUnityCompartment
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 This model is exactly the same as model 1206, but the species is interpreted as an amount, not a concentration, and that amount is explicitly set to change by zero, instead of that change being implied as in 1206.

The model contains:
* 1 species (S1)
* 1 parameter (x)
* 1 compartment (C1)

There are 3 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | S1 | $0$ |
| Rate | C1 | $0.2$ |
| Assignment | x | $S1 / C1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $3$ | variable |
| Initial value of parameter x | $S1 / C1$ | variable |
| Initial volume of compartment 'C1' | $0.5$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

