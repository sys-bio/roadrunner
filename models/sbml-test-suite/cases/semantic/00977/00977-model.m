(*

category:      Test
synopsis:      A three-step reaction scheme with a species conversion factor.
componentTags: Compartment, Parameter, Reaction, Species
testTags:      Amount, ConversionFactors, NonUnityStoichiometry
testType:      TimeCourse
levels:        3.1, 3.2
generatedBy:   Numeric

This model tests conversion factors through a system of three reactions.  There is a conversion factor ('S1conv') on species S1.  This means that S1 is converted according to S1conv, and S2 is not converted.  This changes how S1 change in time, but not how S1 is interpreted in the reaction rates.


The model contains:
* 2 species (S1, S2)
* 5 parameters (k1, k2, k3, S1conv, modelconv)
* 1 compartment (default_compartment)

There are 3 reactions:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| -> S1 | $k1$ |
| 2S1 -> 3S2 | $k2 * S1 / S2$ |
| S2 -> | $k3 * S2$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $0$ | variable |
| Initial concentration of species S2 | $0.001$ | variable |
| Initial value of parameter k1 | $1$ | constant |
| Initial value of parameter k2 | $3$ | constant |
| Initial value of parameter k3 | $1.4$ | constant |
| Initial value of parameter S1conv | $3$ | constant |
| Initial value of parameter modelconv | $4$ | constant |
| Initial volume of compartment 'default_compartment' | $1$ | constant |]

*)

