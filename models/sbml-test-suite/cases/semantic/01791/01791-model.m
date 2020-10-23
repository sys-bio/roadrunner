(*

category:        Test
synopsis:        A compartment or parameter with no 'constant' attribute defaults to 'true'.
componentTags:   AlgebraicRule, Compartment, Parameter
testTags:        DefaultValue, InitialValueReassigned, NonUnityCompartment
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5
generatedBy:     Analytic
packagesPresent: 

An algebraic rule sets the value for non-constant elements in its formula.  Here, one of the two symbols used is a non-constant compartment, and the other is a parameter with no explicitly-set value, meaning that it defaults to constant=true.  This lets C1 change to 3, and leaves P2 fixed at 3, instead of the other way 'round.

An algebraic rule is the only mathematical construct that cares (outside of validation) whether an element is constant or not.

The model contains:
* 1 parameter (P2)
* 1 compartment (C1)

There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Algebraic | $0$ | $P2 - C1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter P2 | $3$ | constant |
| Initial volume of compartment 'C1' | $2$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
