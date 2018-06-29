(*

category:        Test
synopsis:        A parameter with the id 'avogadro'
componentTags:   Compartment, Parameter, Reaction, Species
testTags:        Amount
testType:        TimeCourse
levels:          1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, a parameter named 'avogadro' is used in a kinetic law.

The model contains:
* 1 species (S1)
* 1 parameter (avogadro)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S1 | $avogadro$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $1$ | variable |
| Initial value of parameter avogadro | $0.1$ | constant |
| Initial volume of compartment 'C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
