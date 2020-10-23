(*

category:      Test
synopsis:      Species with an initialConcentration but hasOnlySubstanceUnits="true".
componentTags: Compartment, Reaction, Species
testTags:      Amount, HasOnlySubstanceUnits, NonUnityCompartment
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Numeric

The single species in this model has an initial concentration of '1', but lives in a compartment of size 5, giving it an initial amount of 5. Since 'hasOnlySubstanceUnits' is 'true', this amount is what is used for the rate rule of the reaction (10/S1).

For comparison, if the initialConcentration of the species is changed to an initialAmount of '5', the same numerical results should be obtained.

The model contains:
* 1 species (S1)
* 1 compartment (comp)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| -> S1 | $10 / S1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $5$ | variable |
| Initial volume of compartment 'comp' | $5$ | constant |]

*)

