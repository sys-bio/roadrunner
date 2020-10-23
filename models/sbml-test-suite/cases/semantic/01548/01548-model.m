(*

category:        Test
synopsis:        A fast reaction using a function definition.
componentTags:   Compartment, FunctionDefinition, Reaction, Species
testTags:        Amount, FastReaction
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1
generatedBy:     Analytic
packagesPresent: 

 In this model, the kinetic law of a fast reaction is negative because of a function definition.

The model contains:
* 2 species (S1, S2)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |  *Fast*  |
| J0: S1 -> S2 | $S2 * isneg()$ | fast |]

The model contains the following function definition:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | isneg |  | $-0.01$ |
]
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $2$ | variable |
| Initial amount of species S2 | $3$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
