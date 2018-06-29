(*

category:        Test
synopsis:        Model conversion factor for species only affected by rate rules.
componentTags:   Compartment, Parameter, RateRule, Species
testTags:        Amount, ConversionFactors
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, a model conversion factor is set for species, but those species only change due to rate rules, which means the conversion factor is not used (as they only affect reactions).

The model contains:
* 2 species (S1, S2)
* 2 parameters (m_cf, s1_cf)
* 1 compartment (C)
The conversion factor for the model is 'm_cf'
The conversion factor for the species 'S1' is 's1_cf'


There are 2 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | S1 | $-0.01$ |
| Rate | S2 | $0.01$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $2$ | variable |
| Initial amount of species S2 | $3$ | variable |
| Initial value of parameter m_cf | $3$ | constant |
| Initial value of parameter s1_cf | $5$ | constant |
| Initial volume of compartment 'C' | $1$ | constant |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
