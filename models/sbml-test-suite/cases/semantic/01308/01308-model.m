(*

category:        Test
synopsis:        A species always used as amount, with requested output in concentration
componentTags:   Compartment, RateRule, Species
testTags:        Concentration, HasOnlySubstanceUnits, NonUnityCompartment
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this simple model, a species whose level is always used in terms of amount in the model (initialAmount, hasOnlySubstanceUnits=true) has requested output in terms of concentration.  This is more a test of the test suite system itself than a modeling test per se.

The model contains:
* 1 species (S1)
* 1 compartment (C)

There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | S1 | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $2$ | variable |
| Initial volume of compartment 'C' | $3$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
