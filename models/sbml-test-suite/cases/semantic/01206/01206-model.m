(*

category:        Test
synopsis:        Constant-amount species in variable compartment.
componentTags:   AssignmentRule, Compartment, Parameter, RateRule, Species
testTags:        Amount, InitialValueReassigned, NonConstantCompartment, NonConstantParameter, NonUnityCompartment
testType:        TimeCourse
levels:          1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, the compartment size varies, making the species concentration change (which is assigned to 'x'), but the species amount does not.

The model contains:
* 1 species (S1)
* 1 parameter (x)
* 1 compartment (C1)

There are 2 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | C1 | $0.2$ |
| Assignment | x | $S1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $3$ | variable |
| Initial value of parameter x | $S1$ | variable |
| Initial volume of compartment 'C1' | $0.5$ | variable |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

