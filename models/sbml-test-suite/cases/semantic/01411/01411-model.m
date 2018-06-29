(*

category:        Test
synopsis:        A delay with a function definition as an argument
componentTags:   AssignmentRule, CSymbolDelay, Compartment, FunctionDefinition, Parameter, Reaction, Species
testTags:        Amount, InitialValueReassigned, NonConstantParameter, NonUnityStoichiometry
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Numeric
packagesPresent: 

 In this model, a delay function takes a function definition as an argument.

The model contains:
* 2 species (S1, S2)
* 1 parameter (P1)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: S1 -> 2S2 | $0.1 * S1$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | P1 | $delay(addtwo(S1, S2), 1)$ |]

The model contains the following function definition:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | addtwo | a, b | $a + b$ |
]
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $5$ | variable |
| Initial concentration of species S2 | $1$ | variable |
| Initial value of parameter P1 | $delay(addtwo(S1, S2), 1)$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

*)
