(*

category:        Test
synopsis:        The use of unusual MathML constructs assigned to stoichiometries.
componentTags:   Compartment, InitialAssignment, Reaction, Species
testTags:        Amount, AssignedConstantStoichiometry, InitialValueReassigned, NonUnityStoichiometry, UncommonMathML
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 The model tests various unusual MathML elements in stoichiometries.  It is numerically identical to model 01515, which uses rate rules.

The model contains:
* 38 species (P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20, P21, P22, P23, P24, P25, P26, P27, P28, P29, P30, P31, P32, P33, P34, P35, P36, P37, P38, P39)
* 1 compartment (C)
* 38 species references (P2_sr, P3_sr, P4_sr, P5_sr, P6_sr, P7_sr, P8_sr, P9_sr, P10_sr, P11_sr, P12_sr, P13_sr, P14_sr, P15_sr, P16_sr, P17_sr, P18_sr, P19_sr, P20_sr, P21_sr, P22_sr, P23_sr, P24_sr, P25_sr, P26_sr, P27_sr, P28_sr, P29_sr, P30_sr, P31_sr, P32_sr, P33_sr, P34_sr, P35_sr, P36_sr, P37_sr, P38_sr, P39_sr)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: -> P2_sr P2 + P3_sr P3 + P4_sr P4 + P5_sr P5 + P6_sr P6 + P7_sr P7 + P8_sr P8 + P9_sr P9 + P10_sr P10 + P11_sr P11 + P12_sr P12 + P13_sr P13 + P14_sr P14 + P15_sr P15 + P16_sr P16 + P17_sr P17 + P18_sr P18 + P19_sr P19 + P20_sr P20 + P21_sr P21 + P22_sr P22 + P23_sr P23 + P24_sr P24 + P25_sr P25 + P26_sr P26 + P27_sr P27 + P28_sr P28 + P29_sr P29 + P30_sr P30 + P31_sr P31 + P32_sr P32 + P33_sr P33 + P34_sr P34 + P35_sr P35 + P36_sr P36 + P37_sr P37 + P38_sr P38 + P39_sr P39 | $1$ |]
Note:  the following stoichiometries are set separately:  P2_sr, P3_sr, P4_sr, P5_sr, P6_sr, P7_sr, P8_sr, P9_sr, P10_sr, P11_sr, P12_sr, P13_sr, P14_sr, P15_sr, P16_sr, P17_sr, P18_sr, P19_sr, P20_sr, P21_sr, P22_sr, P23_sr, P24_sr, P25_sr, P26_sr, P27_sr, P28_sr, P29_sr, P30_sr, P31_sr, P32_sr, P33_sr, P34_sr, P35_sr, P36_sr, P37_sr, P38_sr, P39_sr

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species P2 | $0$ | variable |
| Initial concentration of species P3 | $0$ | variable |
| Initial concentration of species P4 | $0$ | variable |
| Initial concentration of species P5 | $0$ | variable |
| Initial concentration of species P6 | $0$ | variable |
| Initial concentration of species P7 | $0$ | variable |
| Initial concentration of species P8 | $0$ | variable |
| Initial concentration of species P9 | $0$ | variable |
| Initial concentration of species P10 | $0$ | variable |
| Initial concentration of species P11 | $0$ | variable |
| Initial concentration of species P12 | $0$ | variable |
| Initial concentration of species P13 | $0$ | variable |
| Initial concentration of species P14 | $0$ | variable |
| Initial concentration of species P15 | $0$ | variable |
| Initial concentration of species P16 | $0$ | variable |
| Initial concentration of species P17 | $0$ | variable |
| Initial concentration of species P18 | $0$ | variable |
| Initial concentration of species P19 | $0$ | variable |
| Initial concentration of species P20 | $0$ | variable |
| Initial concentration of species P21 | $0$ | variable |
| Initial concentration of species P22 | $0$ | variable |
| Initial concentration of species P23 | $0$ | variable |
| Initial concentration of species P24 | $0$ | variable |
| Initial concentration of species P25 | $0$ | variable |
| Initial concentration of species P26 | $0$ | variable |
| Initial concentration of species P27 | $0$ | variable |
| Initial concentration of species P28 | $0$ | variable |
| Initial concentration of species P29 | $0$ | variable |
| Initial concentration of species P30 | $0$ | variable |
| Initial concentration of species P31 | $0$ | variable |
| Initial concentration of species P32 | $0$ | variable |
| Initial concentration of species P33 | $0$ | variable |
| Initial concentration of species P34 | $0$ | variable |
| Initial concentration of species P35 | $0$ | variable |
| Initial concentration of species P36 | $0$ | variable |
| Initial concentration of species P37 | $0$ | variable |
| Initial concentration of species P38 | $0$ | variable |
| Initial concentration of species P39 | $0$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |
| Initial value of species reference 'P2_sr' | $abs(-1)$ | constant |
| Initial value of species reference 'P3_sr' | $abs(1)$ | constant |
| Initial value of species reference 'P4_sr' | $acos(-1)$ | constant |
| Initial value of species reference 'P5_sr' | $acos(0.5)$ | constant |
| Initial value of species reference 'P6_sr' | $asin(1)$ | constant |
| Initial value of species reference 'P7_sr' | $asin(-0.5)$ | constant |
| Initial value of species reference 'P8_sr' | $atan(2.8)$ | constant |
| Initial value of species reference 'P9_sr' | $atan(-7.09)$ | constant |
| Initial value of species reference 'P10_sr' | $ceil(0.5)$ | constant |
| Initial value of species reference 'P11_sr' | $ceil(3.55)$ | constant |
| Initial value of species reference 'P12_sr' | $ceil(-4.6)$ | constant |
| Initial value of species reference 'P13_sr' | $cos(9.1)$ | constant |
| Initial value of species reference 'P14_sr' | $cos(-0.22)$ | constant |
| Initial value of species reference 'P15_sr' | $exp(0)$ | constant |
| Initial value of species reference 'P16_sr' | $exp(1)$ | constant |
| Initial value of species reference 'P17_sr' | $exp(0.77)$ | constant |
| Initial value of species reference 'P18_sr' | $floor(-4.6)$ | constant |
| Initial value of species reference 'P19_sr' | $floor(9.1)$ | constant |
| Initial value of species reference 'P20_sr' | $ln(0.2)$ | constant |
| Initial value of species reference 'P21_sr' | $ln(1)$ | constant |
| Initial value of species reference 'P22_sr' | $log10(0.2)$ | constant |
| Initial value of species reference 'P23_sr' | $log10(1)$ | constant |
| Initial value of species reference 'P24_sr' | $1^2$ | constant |
| Initial value of species reference 'P25_sr' | $2^2$ | constant |
| Initial value of species reference 'P26_sr' | $1.4^5.1$ | constant |
| Initial value of species reference 'P27_sr' | $4^2$ | constant |
| Initial value of species reference 'P28_sr' | $3.1^2$ | constant |
| Initial value of species reference 'P29_sr' | $sqrt(4)$ | constant |
| Initial value of species reference 'P30_sr' | $sqrt(7.4)$ | constant |
| Initial value of species reference 'P31_sr' | $sin(2.1)$ | constant |
| Initial value of species reference 'P32_sr' | $sin(0)$ | constant |
| Initial value of species reference 'P33_sr' | $sin(-5.9)$ | constant |
| Initial value of species reference 'P34_sr' | $tan(0)$ | constant |
| Initial value of species reference 'P35_sr' | $tan(1.11)$ | constant |
| Initial value of species reference 'P36_sr' | $tan(-6)$ | constant |
| Initial value of species reference 'P37_sr' | $tanh(0)$ | constant |
| Initial value of species reference 'P38_sr' | $tanh(1.11)$ | constant |
| Initial value of species reference 'P39_sr' | $tanh(-6)$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
