(*

category:      Test
synopsis:      A species that with an initial concentration set, and subsequently produced by a reaction, being echoed by a parameter viewing it through a delay.
componentTags: AssignmentRule, CSymbolDelay, Compartment, Parameter, Reaction, Species
testTags:      Amount, NonConstantParameter, InitialValueReassigned
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

 The model contains a single species x with an initial concentration, which at the start of the simulation increases due to a reaction.  A parameter (y) echoes this species with a 0.2 second delay, meaning that it starts by 'seeing' the initial constant value before echoing the linear increase.


The model contains:
* 1 species (x)
* 1 parameter (y)
* 1 compartment (default_compartment)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| -> x | $1$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | y | $2 + delay(x, 0.2)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species x | $3$ | variable |
| Initial value of parameter y | $2 + delay(x, 0.2)$ | variable |
| Initial volume of compartment 'default_compartment' | $1$ | constant |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

