(*

category:        Test
synopsis:        A compartment and species with no 'constant' attributes.
componentTags:   AlgebraicRule, Compartment, Parameter, Species
testTags:        Amount, DefaultValue, HasOnlySubstanceUnits, InitialValueReassigned
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5
generatedBy:     Analytic
packagesPresent: 

An algebraic rule sets the value for non-constant elements in its formula.  Here, one of the two symbols used is a parameter with no explicitly-set 'constant', which defalts to 'true', and the other is a species with no explicitly-set 'constant' value, meaning that it defaults to constant=false.  This lets S2 change to 2, and leaves P1 fixed at 2, instead of the other way 'round.

An algebraic rule is the only mathematical construct that cares (outside of validation) whether an element is constant or not.

The model contains:
* 1 species (S2)
* 1 parameter (P1)
* 1 compartment (C)

There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Algebraic | $0$ | $P1 - S2$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S2 | $3$ | variable |
| Initial value of parameter P1 | $2$ | constant |
| Initial volume of compartment 'C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
