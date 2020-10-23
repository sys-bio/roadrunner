(*

category:        Test
synopsis:        A species and a compartment both under control of a rate rule.
componentTags:   AssignmentRule, CSymbolRateOf, Compartment, Parameter, RateRule, Species
testTags:        Amount, BoundaryCondition, HasOnlySubstanceUnits, InitialValueReassigned, NonConstantCompartment, NonConstantParameter, NonUnityCompartment
testType:        TimeCourse
levels:          3.2
generatedBy:     Numeric
packagesPresent: 

 This test modifies test 1462 slightly so that the species is set 'hasOnlySubstanceUnits=true', meaning that the species concentration is affected more straightforwardly by the two rates of change.  This test checks to ensure that the rateOf S1 continues to be correctly reported (this time as the rate of change of its amount).

The model contains:
* 1 species (S1)
* 1 parameter (x)
* 1 compartment (C1)

There are 3 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | S1 | $0.4$ |
| Rate | C1 | $0.2$ |
| Assignment | x | $rateOf(S1)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $0$ | variable |
| Initial value of parameter x | $rateOf(S1)$ | variable |
| Initial volume of compartment 'C1' | $0.5$ | variable |]

*)
