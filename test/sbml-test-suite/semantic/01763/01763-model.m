(*

category:        Test
synopsis:        A parameter named 'avogadro' as well as the csymbol avogadro.
componentTags:   CSymbolAvogadro, Compartment, Parameter, Reaction, Species
testTags:        Amount
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, a parameter is given the id 'avogadro', and the kinetic law references both it and the avogadro csymbol.

The model contains:
* 1 species (S1)
* 1 parameter (avogadro)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S1 | $avogadro / avogadro$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $1$ | variable |
| Initial value of parameter avogadro | $1e+24$ | constant |
| Initial volume of compartment 'C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
