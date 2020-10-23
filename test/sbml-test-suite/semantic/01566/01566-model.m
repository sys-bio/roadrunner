(*

category:        Test
synopsis:        Uncommon MathML that uses species references.
componentTags:   AssignmentRule, Compartment, Parameter, Reaction, Species
testTags:        Amount, InitialValueReassigned, NonUnityStoichiometry, SpeciesReferenceInMath, UncommonMathML
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

The model tests various uncommon mathematical constructs using a species reference.

The model contains:
* 1 species (S1)
* 49 parameters (P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20, P21, P22, P23, P24, P25, P26, P27, P28, P29, P30, P31, P32, P33, P34, P35, P36, P37, P38, P39, P40, P41, P42, P43, P44, P45, P46, P47, P48, P49)
* 1 compartment (C)
* 1 species reference (S1_stoich)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> 2S1 | $1$ |]


There are 49 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | P1 | $exp(S1_stoich)$ |
| Assignment | P2 | $abs(-S1_stoich)$ |
| Assignment | P3 | $abs(S1_stoich)$ |
| Assignment | P4 | $acos(-S1_stoich + 1)$ |
| Assignment | P5 | $acos(S1_stoich / 4)$ |
| Assignment | P6 | $asin(S1_stoich - 1)$ |
| Assignment | P7 | $asin(-S1_stoich / 4)$ |
| Assignment | P8 | $atan(S1_stoich + 0.8)$ |
| Assignment | P9 | $atan(-S1_stoich * 3 - 1.09)$ |
| Assignment | P10 | $ceil(S1_stoich / 4)$ |
| Assignment | P11 | $ceil(S1_stoich * 4 - 0.45)$ |
| Assignment | P12 | $ceil(-S1_stoich * 2 - 0.6)$ |
| Assignment | P13 | $cos(S1_stoich * 4 + 1.1)$ |
| Assignment | P14 | $cos(-S1_stoich / 10 - 0.02)$ |
| Assignment | P15 | $exp(0 * S1_stoich)$ |
| Assignment | P16 | $exp(S1_stoich / 2)$ |
| Assignment | P17 | $exp(S1_stoich / 2 - 0.33)$ |
| Assignment | P18 | $floor(-S1_stoich * 2 - 0.6)$ |
| Assignment | P19 | $floor(S1_stoich * 4 + 1.1)$ |
| Assignment | P20 | $ln(S1_stoich / 10)$ |
| Assignment | P21 | $ln(S1_stoich / 2)$ |
| Assignment | P22 | $log10(S1_stoich / 10)$ |
| Assignment | P23 | $log10(S1_stoich / 2)$ |
| Assignment | P24 | $1^S1_stoich$ |
| Assignment | P25 | $S1_stoich^S1_stoich$ |
| Assignment | P26 | $1.4^5.1$ |
| Assignment | P27 | $(S1_stoich * S1_stoich)^S1_stoich$ |
| Assignment | P28 | $3.1^S1_stoich$ |
| Assignment | P29 | $sqrt(S1_stoich * 2)$ |
| Assignment | P30 | $sqrt(7 + S1_stoich / 5)$ |
| Assignment | P31 | $sin(S1_stoich + 0.1)$ |
| Assignment | P32 | $sin(0 * S1_stoich)$ |
| Assignment | P33 | $sin(-S1_stoich * 2 - 1.9)$ |
| Assignment | P34 | $tan(0 * S1_stoich)$ |
| Assignment | P35 | $tan(S1_stoich / 2 + 0.11)$ |
| Assignment | P36 | $tan(-S1_stoich * 3)$ |
| Assignment | P37 | $sec(S1_stoich / 4)$ |
| Assignment | P38 | $csc(S1_stoich * 2.25)$ |
| Assignment | P39 | $cot(S1_stoich / 10)$ |
| Assignment | P40 | $sinh(S1_stoich / 10 + 0.1)$ |
| Assignment | P41 | $cosh(S1_stoich - 0.3)$ |
| Assignment | P42 | $arcsec(S1_stoich + 0.3)$ |
| Assignment | P43 | $arccsc(S1_stoich - 0.9)$ |
| Assignment | P44 | $arccot(S1_stoich - 2.1)$ |
| Assignment | P45 | $arcsinh(S1_stoich * 50 - 1)$ |
| Assignment | P46 | $arccosh(S1_stoich / 2 + 0.34)$ |
| Assignment | P47 | $arctanh(S1_stoich - 2.7)$ |
| Assignment | P48 | $arcsech(S1_stoich * 0.21)$ |
| Assignment | P49 | $arccsch(S1_stoich * 0.005)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species S1 | $0$ | variable |
| Initial value of parameter P1 | $exp(S1_stoich)$ | variable |
| Initial value of parameter P2 | $abs(-S1_stoich)$ | variable |
| Initial value of parameter P3 | $abs(S1_stoich)$ | variable |
| Initial value of parameter P4 | $acos(-S1_stoich + 1)$ | variable |
| Initial value of parameter P5 | $acos(S1_stoich / 4)$ | variable |
| Initial value of parameter P6 | $asin(S1_stoich - 1)$ | variable |
| Initial value of parameter P7 | $asin(-S1_stoich / 4)$ | variable |
| Initial value of parameter P8 | $atan(S1_stoich + 0.8)$ | variable |
| Initial value of parameter P9 | $atan(-S1_stoich * 3 - 1.09)$ | variable |
| Initial value of parameter P10 | $ceil(S1_stoich / 4)$ | variable |
| Initial value of parameter P11 | $ceil(S1_stoich * 4 - 0.45)$ | variable |
| Initial value of parameter P12 | $ceil(-S1_stoich * 2 - 0.6)$ | variable |
| Initial value of parameter P13 | $cos(S1_stoich * 4 + 1.1)$ | variable |
| Initial value of parameter P14 | $cos(-S1_stoich / 10 - 0.02)$ | variable |
| Initial value of parameter P15 | $exp(0 * S1_stoich)$ | variable |
| Initial value of parameter P16 | $exp(S1_stoich / 2)$ | variable |
| Initial value of parameter P17 | $exp(S1_stoich / 2 - 0.33)$ | variable |
| Initial value of parameter P18 | $floor(-S1_stoich * 2 - 0.6)$ | variable |
| Initial value of parameter P19 | $floor(S1_stoich * 4 + 1.1)$ | variable |
| Initial value of parameter P20 | $ln(S1_stoich / 10)$ | variable |
| Initial value of parameter P21 | $ln(S1_stoich / 2)$ | variable |
| Initial value of parameter P22 | $log10(S1_stoich / 10)$ | variable |
| Initial value of parameter P23 | $log10(S1_stoich / 2)$ | variable |
| Initial value of parameter P24 | $1^S1_stoich$ | variable |
| Initial value of parameter P25 | $S1_stoich^S1_stoich$ | variable |
| Initial value of parameter P26 | $1.4^5.1$ | variable |
| Initial value of parameter P27 | $(S1_stoich * S1_stoich)^S1_stoich$ | variable |
| Initial value of parameter P28 | $3.1^S1_stoich$ | variable |
| Initial value of parameter P29 | $sqrt(S1_stoich * 2)$ | variable |
| Initial value of parameter P30 | $sqrt(7 + S1_stoich / 5)$ | variable |
| Initial value of parameter P31 | $sin(S1_stoich + 0.1)$ | variable |
| Initial value of parameter P32 | $sin(0 * S1_stoich)$ | variable |
| Initial value of parameter P33 | $sin(-S1_stoich * 2 - 1.9)$ | variable |
| Initial value of parameter P34 | $tan(0 * S1_stoich)$ | variable |
| Initial value of parameter P35 | $tan(S1_stoich / 2 + 0.11)$ | variable |
| Initial value of parameter P36 | $tan(-S1_stoich * 3)$ | variable |
| Initial value of parameter P37 | $sec(S1_stoich / 4)$ | variable |
| Initial value of parameter P38 | $csc(S1_stoich * 2.25)$ | variable |
| Initial value of parameter P39 | $cot(S1_stoich / 10)$ | variable |
| Initial value of parameter P40 | $sinh(S1_stoich / 10 + 0.1)$ | variable |
| Initial value of parameter P41 | $cosh(S1_stoich - 0.3)$ | variable |
| Initial value of parameter P42 | $arcsec(S1_stoich + 0.3)$ | variable |
| Initial value of parameter P43 | $arccsc(S1_stoich - 0.9)$ | variable |
| Initial value of parameter P44 | $arccot(S1_stoich - 2.1)$ | variable |
| Initial value of parameter P45 | $arcsinh(S1_stoich * 50 - 1)$ | variable |
| Initial value of parameter P46 | $arccosh(S1_stoich / 2 + 0.34)$ | variable |
| Initial value of parameter P47 | $arctanh(S1_stoich - 2.7)$ | variable |
| Initial value of parameter P48 | $arcsech(S1_stoich * 0.21)$ | variable |
| Initial value of parameter P49 | $arccsch(S1_stoich * 0.005)$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |
| Initial value of species reference 'S1_stoich' | $2$ | constant |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
