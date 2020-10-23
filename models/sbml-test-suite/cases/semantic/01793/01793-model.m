(*

category:        Test
synopsis:        A compartment and species with no 'constant' attributes.
componentTags:   AlgebraicRule, Compartment, Species
testTags:        Amount, DefaultValue, HasOnlySubstanceUnits, InitialValueReassigned
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5
generatedBy:     Analytic
packagesPresent: 

An algebraic rule sets the value for non-constant elements in its formula.  Here, one of the two symbols used is a compartment with no explicitly-set 'constant', which defalts to 'true', and the other is a species with no explicitly-set 'constant' value, meaning that it defaults to constant=false.  This lets S2 change to 1, and leaves C1 fixed at 1, instead of the other way 'round.

An algebraic rule is the only mathematical construct that cares (outside of validation) whether an element is constant or not.

The model contains:
* 1 species (S2)
* 1 compartment (C1)

There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Algebraic | $0$ | $1 * C1 - S2$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S2 | $3$ | variable |
| Initial volume of compartment 'C1' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
