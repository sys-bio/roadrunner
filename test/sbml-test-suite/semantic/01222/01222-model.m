(*

category:        Test
synopsis:        A boundary substance-only species in varying compartment under control of a rate rule.
componentTags:   CSymbolTime, Compartment, EventNoDelay, RateRule, Species
testTags:        Amount, BoundaryCondition, HasOnlySubstanceUnits, NonConstantCompartment, NonUnityCompartment
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, a boundary species with 'hasOnlySubstanceUnits=true' is changing according to a rate rule, and is in a compartment which itself changes due to an event.  However, since the species is always described in terms of amount, the fact that its compartment changes does not affect it.  This test ensures that simulators apply the rate rule to the species amount, and not to the species concentration.

The model contains:
* 1 species (S1)
* 1 compartment (c)

There is one event:

[{width:30em,margin: 1em auto}|  *Event*  |  *Trigger*  | *Event Assignments* |
| _E0 | $time > 5.5$ | $c = 4$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | S1 | $1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $1.2$ | variable |
| Initial volume of compartment 'c' | $2$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

