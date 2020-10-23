(*

category:        Test
synopsis:        The rateOf csymbol applied to a local parameter.
componentTags:   CSymbolRateOf, Compartment, Parameter, RateRule, Reaction, Species
testTags:        Amount, LocalParameters, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Analytic
packagesPresent: 

 The kinetic law in this test takes the rateOf csymbol of a local parameter, which, by definition, must be zero, since local parameters may not change.  There is an additional global parameter with the same ID that *does* change in time,which must not be used by the rateOf function.

The model contains:
* 1 species (S1)
* 1 parameter (k1)
* 1 local parameter (J0.k1)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S1 | $rateOf(k1)$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | k1 | $2$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $1$ | variable |
| Initial value of parameter k1 | $1$ | variable |
| Initial value of local parameter 'J0.k1' | $3$ | constant |
| Initial volume of compartment 'C' | $1$ | constant |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
