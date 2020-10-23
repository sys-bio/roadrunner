(*

category:        Test
synopsis:        A species with no 'constant' attribute defaults to 'false'.
componentTags:   AlgebraicRule, Compartment, Species
testTags:        Amount, BoundaryCondition, DefaultValue, InitialValueReassigned
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5
generatedBy:     Analytic
packagesPresent: 

An algebraic rule sets the value for non-constant elements in its formula.  Here, one of the two symbols used is a constant species, and the other is a species with no explicitly-set value, meaning that it defaults to constant=false.  This lets S1 change to 3, and leaves S2 fixed at 3, instead of the other way 'round.An algebraic rule is the only mathematical construct that cares (outside of validation) whether an element is constant or not.

The model contains:
* 2 species (S1, S2)
* 1 compartment (C)

There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Algebraic | $0$ | $S2 - S1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $2$ | variable |
| Initial concentration of species S2 | $3$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
