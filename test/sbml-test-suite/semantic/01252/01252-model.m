(*

category:        Test
synopsis:        The rateOf csymbol for a species that changes due to a reaction.
componentTags:   CSymbolRateOf, Compartment, InitialAssignment, Parameter, Reaction, Species
testTags:        Amount, InitialValueReassigned
testType:        TimeCourse
levels:          3.2
generatedBy:     Analytic
packagesPresent: 

 The rateOf csymbol is used here in an initial assignment that references a species changing due to a reaction, and thus takes the value of the reaction's kinetic law, since it's a product.

The model contains:
* 1 species (S1)
* 1 parameter (p2)
* 1 compartment (c)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S1 | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $100$ | variable |
| Initial value of parameter p2 | $rateOf(S1)$ | variable |
| Initial volume of compartment 'c' | $1$ | constant |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
