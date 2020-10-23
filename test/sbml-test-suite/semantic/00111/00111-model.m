(*

category:        Test
synopsis:        Reactions occurring between two compartments with functionDefinitions.
componentTags:   Compartment, FunctionDefinition, Parameter, Reaction, Species
testTags:        Amount, MultiCompartment, NonUnityStoichiometry
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Numeric
packagesPresent: 

The model contains two compartments named compartment and compartment1.
There are four species named S1, S2, S3 and S4 and three parameters named
k1, k2 and k3.  Compartment "compartment" contains species S1 and S2.
Compartment "compartment1" contains species S3 and S4.

The model contains:
* 4 species (S1, S2, S3, S4)
* 3 parameters (k1, k2, k3)
* 2 compartments (compartment, compartment1)

There are 3 reactions:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| reaction1: S1 + S2 -> 2S2 | $compartment * k1 * multiply(S1, S2)$ |
| reaction2: S2 -> S3 | $compartment1 * k2 * (S2 + -1 * S3)$ |
| reaction3: S3 + S4 -> 2S4 | $compartment * k3 * S3 * S4$ |]

The first reaction occurs entirely within compartment "compartment", the
second reaction occurs between a species in "compartment" and a species in
"compartment1", and the third reaction occurs entirely within
"compartment1".

The model contains the following function definition:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | multiply | x, y | $x * y$ |
]
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $1$ | variable |
| Initial amount of species S2 | $1$ | variable |
| Initial amount of species S3 | $0$ | variable |
| Initial amount of species S4 | $0.1$ | variable |
| Initial value of parameter k1 | $0.75$ | constant |
| Initial value of parameter k2 | $7.5$ | constant |
| Initial value of parameter k3 | $0.75$ | constant |
| Initial volume of compartment 'compartment' | $1$ | constant |
| Initial volume of compartment 'compartment1' | $1$ | constant |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

*)

newcase[ "00111" ];

addFunction[ multiply, arguments -> {x, y}, math -> x * y];
addCompartment[ compartment, size -> 1 ];
addCompartment[ compartment1, size -> 1 ];
addSpecies[ S1, compartment->compartment, initialAmount -> 1.0];
addSpecies[ S2, compartment->compartment, initialAmount -> 1.0];
addSpecies[ S3, compartment->compartment1, initialAmount -> 0];
addSpecies[ S4, compartment->compartment1, initialAmount -> 0.1];
addParameter[ k1, value -> 0.75 ];
addParameter[ k2, value -> 7.5 ];
addParameter[ k3, value -> 0.75 ];
addReaction[ S1 + S2 -> S2  +  S2, reversible -> False,
	     kineticLaw -> k1 * multiply[S1,S2] * compartment ];
addReaction[ S2 -> S3, reversible -> False,
	     kineticLaw -> k2 * (S2-S3) * compartment1 ];
addReaction[ S3 + S4 -> S4  +  S4, reversible -> False,
	     kineticLaw -> k3 * S3 * S4 * compartment ];

makemodel[]

