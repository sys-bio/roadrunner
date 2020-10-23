(*

category:        Test
synopsis:        A compartment with no 'constant' attribute defaults to 'true'.
componentTags:   AlgebraicRule, Compartment
testTags:        DefaultValue, InitialValueReassigned, MultiCompartment, NonUnityCompartment
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5
generatedBy:     Analytic
packagesPresent: 

An algebraic rule sets the value for non-constant elements in its formula.  Here, one of the two symbols used is a non-constant compartment, and the other is a compartment with no explicitly-set value, meaning that it defaults to constant=true.  This lets C2 change to 2, and leaves C1 fixed at 2, instead of the other way 'round.An algebraic rule is the only mathematical construct that cares (outside of validation) whether an element is constant or not.  This model is identical to model 17865, except which compartment is constant is reversed, to ensure that simulators don't just use 'the first compartment' or a similar rule for resolving ambiguities like this.

The model contains:
* 2 compartments (C1, C2)

There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Algebraic | $0$ | $C2 - C1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial volume of compartment 'C1' | $2$ | constant |
| Initial volume of compartment 'C2' | $3$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
