(*

category:        Test
synopsis:        A combination of a model converstion factor and the rateOf function.
componentTags:   AssignmentRule, CSymbolRateOf, Compartment, Parameter, Reaction, Species
testTags:        Amount, ConversionFactors, InitialValueReassigned, NonConstantParameter
testType:        TimeCourse
levels:          3.2
generatedBy:     Numeric
packagesPresent: 

 In this model, the species are affected by a reaction, which is itself affected by a model-wide conversion factor.  The parameter P0 ensures that the rateOf function incorporates the conversion factor into its own return value.

The model contains:
* 2 species (S1, S2)
* 2 parameters (P0, conversion)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: S1 -> S2 | $0.1 * S1$ |]

The conversion factor for the model is 'conversion'


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | P0 | $rateOf(S1)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $5$ | variable |
| Initial concentration of species S2 | $1$ | variable |
| Initial value of parameter conversion | $2$ | constant |
| Initial value of parameter P0 | $rateOf(S1)$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

*)
