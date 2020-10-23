(*

category:        Test
synopsis:        Variable compartment with variable species concentration
componentTags:   AssignmentRule, Compartment, Parameter, RateRule, Species
testTags:        Amount, BoundaryCondition, InitialValueReassigned, NonConstantCompartment, NonConstantParameter, NonUnityCompartment, VolumeConcentrationRates
testType:        TimeCourse
levels:          1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 This model ends up exactly the same as model 1206 and 1207, with a variable compartment, and an explicit rate of change of the species concentration that ends up with the species amount not changing.

If A is the amount of S1, and V is the volume:
d(A/V)/dt = [V dA/dt - A dV/dt]/V^2
          = -A/V^2 dV/dt
          = -A/V * .2 / V
          = -.2*A/V / V
          = -.2*S1/V
 
The model contains:
* 1 species (S1)
* 1 parameter (x)
* 1 compartment (C1)

There are 3 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | S1 | $(-0.2 * S1) / C1$ |
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

