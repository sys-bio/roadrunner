(*

category:        Test
synopsis:        Test that L1 and L2 reactions default to fast=false.
componentTags:   Compartment, Reaction, Species
testTags:        Amount, DefaultValue
testType:        TimeCourse
levels:          1.2, 2.1, 2.2, 2.3, 2.4, 2.5
generatedBy:     Numeric
packagesPresent: 

In this very simple model, a reaction is present that, if it was set 'fast=true', would immediately go to 1.0, but goes there asymptotically instead.  This tests to ensure that in L2 and L1 model reactions with no explicit 'fast' flag, the default value of 'false' is used.

The model contains:
* 1 species (S1)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> S1 | $1 * (1 - S1)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $0$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

*)
