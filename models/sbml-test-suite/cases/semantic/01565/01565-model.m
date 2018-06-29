(*

category:        Test
synopsis:        Uncommon MathML in kinetic laws from fast reactions.
componentTags:   Compartment, Reaction, Species
testTags:        Amount, FastReaction, MultipleFastReactions, UncommonMathML
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1
generatedBy:     Analytic
packagesPresent: 

The model tests various uncommon mathematical constructs in kinetic laws from fast reactions.

The model contains:
* 104 species (S0, S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17, S18, S19, S20, S21, S22, S23, S24, S25, S26, S27, S28, S29, S30, S31, S32, S33, S34, S35, S36, S37, S38, S39, S40, S41, S42, S43, S44, S45, S46, S47, S48, S49, S50, S51, S0b, S1b, S2b, S3b, S4b, S5b, S6b, S7b, S8b, S9b, S10b, S11b, S12b, S13b, S14b, S15b, S16b, S17b, S18b, S19b, S20b, S21b, S22b, S23b, S24b, S25b, S26b, S27b, S28b, S29b, S30b, S31b, S32b, S33b, S34b, S35b, S36b, S37b, S38b, S39b, S40b, S41b, S42b, S43b, S44b, S45b, S46b, S47b, S48b, S49b, S50b, S51b)
* 1 compartment (C)

There are 52 reactions:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |  *Fast*  |
| J0: S0b -> S0 | $1 * S0 * S0b * exponentiale$ | fast |
| J1: S1b -> S1 | $1 * S1 * S1b * exp(exponentiale)$ | fast |
| J2: S2b -> S2 | $1 * S2 * S2b * abs(-1)$ | fast |
| J3: S3b -> S3 | $1 * S3 * S3b * abs(1)$ | fast |
| J4: S4b -> S4 | $1 * S4 * S4b * acos(-1)$ | fast |
| J5: S5b -> S5 | $1 * S5 * S5b * acos(0.5)$ | fast |
| J6: S6b -> S6 | $1 * S6 * S6b * asin(1)$ | fast |
| J7: S7b -> S7 | $1 * S7 * S7b * asin(-0.5)$ | fast |
| J8: S8b -> S8 | $1 * S8 * S8b * atan(2.8)$ | fast |
| J9: S9b -> S9 | $1 * S9 * S9b * atan(-7.09)$ | fast |
| J10: S10b -> S10 | $1 * S10 * S10b * ceil(0.5)$ | fast |
| J11: S11b -> S11 | $1 * S11 * S11b * ceil(3.55)$ | fast |
| J12: S12b -> S12 | $1 * S12 * S12b * ceil(-4.6)$ | fast |
| J13: S13b -> S13 | $1 * S13 * S13b * cos(9.1)$ | fast |
| J14: S14b -> S14 | $1 * S14 * S14b * cos(-0.22)$ | fast |
| J15: S15b -> S15 | $1 * S15 * S15b * exp(0)$ | fast |
| J16: S16b -> S16 | $1 * S16 * S16b * exp(1)$ | fast |
| J17: S17b -> S17 | $1 * S17 * S17b * exp(0.77)$ | fast |
| J18: S18b -> S18 | $1 * S18 * S18b * floor(-4.6)$ | fast |
| J19: S19b -> S19 | $1 * S19 * S19b * floor(9.1)$ | fast |
| J20: S20b -> S20 | $1 * S20 * S20b * ln(0.2)$ | fast |
| J21: S21b -> S21 | $1 * S21 * S21b * ln(1)$ | fast |
| J22: S22b -> S22 | $1 * S22 * S22b * log10(0.2)$ | fast |
| J23: S23b -> S23 | $1 * S23 * S23b * log10(1)$ | fast |
| J24: S24b -> S24 | $1 * S24 * S24b * 1^2$ | fast |
| J25: S25b -> S25 | $1 * S25 * S25b * 2^2$ | fast |
| J26: S26b -> S26 | $1 * S26 * S26b * 1.4^5.1$ | fast |
| J27: S27b -> S27 | $1 * S27 * S27b * 4^2$ | fast |
| J28: S28b -> S28 | $1 * S28 * S28b * 3.1^2$ | fast |
| J29: S29b -> S29 | $1 * S29 * S29b * sqrt(4)$ | fast |
| J30: S30b -> S30 | $1 * S30 * S30b * sqrt(7.4)$ | fast |
| J31: S31b -> S31 | $1 * S31 * S31b * sin(2.1)$ | fast |
| J32: S32b -> S32 | $1 * S32 * S32b * sin(0)$ | fast |
| J33: S33b -> S33 | $1 * S33 * S33b * sin(-5.9)$ | fast |
| J34: S34b -> S34 | $1 * S34 * S34b * tan(0)$ | fast |
| J35: S35b -> S35 | $1 * S35 * S35b * tan(1.11)$ | fast |
| J36: S36b -> S36 | $1 * S36 * S36b * tan(-6)$ | fast |
| J37: S37b -> S37 | $1 * S37 * S37b * sec(0.5)$ | fast |
| J38: S38b -> S38 | $1 * S38 * S38b * csc(4.5)$ | fast |
| J39: S39b -> S39 | $1 * S39 * S39b * cot(0.2)$ | fast |
| J40: S40b -> S40 | $1 * S40 * S40b * sinh(0.3)$ | fast |
| J41: S41b -> S41 | $1 * S41 * S41b * cosh(1.7)$ | fast |
| J42: S42b -> S42 | $1 * S42 * S42b * arcsec(2.3)$ | fast |
| J43: S43b -> S43 | $1 * S43 * S43b * arccsc(1.1)$ | fast |
| J44: S44b -> S44 | $1 * S44 * S44b * arccot(-0.1)$ | fast |
| J45: S45b -> S45 | $1 * S45 * S45b * arcsinh(99)$ | fast |
| J46: S46b -> S46 | $1 * S46 * S46b * arccosh(1.34)$ | fast |
| J47: S47b -> S47 | $1 * S47 * S47b * arctanh(-0.7)$ | fast |
| J48: S48b -> S48 | $1 * S48 * S48b * arcsech(0.42)$ | fast |
| J49: S49b -> S49 | $1 * S49 * S49b * arccsch(0.01)$ | fast |
| J50: S50b -> S50 | $1 * S50 * S50b$ | fast |
| J51: S51b -> S51 | $1 * S51 * S51b * plus()$ | fast |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S0 | $2$ | variable |
| Initial concentration of species S1 | $2$ | variable |
| Initial concentration of species S2 | $2$ | variable |
| Initial concentration of species S3 | $2$ | variable |
| Initial concentration of species S4 | $2$ | variable |
| Initial concentration of species S5 | $2$ | variable |
| Initial concentration of species S6 | $2$ | variable |
| Initial concentration of species S7 | $2$ | variable |
| Initial concentration of species S8 | $2$ | variable |
| Initial concentration of species S9 | $2$ | variable |
| Initial concentration of species S10 | $2$ | variable |
| Initial concentration of species S11 | $2$ | variable |
| Initial concentration of species S12 | $2$ | variable |
| Initial concentration of species S13 | $2$ | variable |
| Initial concentration of species S14 | $2$ | variable |
| Initial concentration of species S15 | $2$ | variable |
| Initial concentration of species S16 | $2$ | variable |
| Initial concentration of species S17 | $2$ | variable |
| Initial concentration of species S18 | $2$ | variable |
| Initial concentration of species S19 | $2$ | variable |
| Initial concentration of species S20 | $2$ | variable |
| Initial concentration of species S21 | $2$ | variable |
| Initial concentration of species S22 | $2$ | variable |
| Initial concentration of species S23 | $2$ | variable |
| Initial concentration of species S24 | $2$ | variable |
| Initial concentration of species S25 | $2$ | variable |
| Initial concentration of species S26 | $2$ | variable |
| Initial concentration of species S27 | $2$ | variable |
| Initial concentration of species S28 | $2$ | variable |
| Initial concentration of species S29 | $2$ | variable |
| Initial concentration of species S30 | $2$ | variable |
| Initial concentration of species S31 | $2$ | variable |
| Initial concentration of species S32 | $2$ | variable |
| Initial concentration of species S33 | $2$ | variable |
| Initial concentration of species S34 | $2$ | variable |
| Initial concentration of species S35 | $2$ | variable |
| Initial concentration of species S36 | $2$ | variable |
| Initial concentration of species S37 | $2$ | variable |
| Initial concentration of species S38 | $2$ | variable |
| Initial concentration of species S39 | $2$ | variable |
| Initial concentration of species S40 | $2$ | variable |
| Initial concentration of species S41 | $2$ | variable |
| Initial concentration of species S42 | $2$ | variable |
| Initial concentration of species S43 | $2$ | variable |
| Initial concentration of species S44 | $2$ | variable |
| Initial concentration of species S45 | $2$ | variable |
| Initial concentration of species S46 | $2$ | variable |
| Initial concentration of species S47 | $2$ | variable |
| Initial concentration of species S48 | $2$ | variable |
| Initial concentration of species S49 | $2$ | variable |
| Initial concentration of species S50 | $2$ | variable |
| Initial concentration of species S51 | $2$ | variable |
| Initial concentration of species S0b | $3$ | variable |
| Initial concentration of species S1b | $3$ | variable |
| Initial concentration of species S2b | $3$ | variable |
| Initial concentration of species S3b | $3$ | variable |
| Initial concentration of species S4b | $3$ | variable |
| Initial concentration of species S5b | $3$ | variable |
| Initial concentration of species S6b | $3$ | variable |
| Initial concentration of species S7b | $3$ | variable |
| Initial concentration of species S8b | $3$ | variable |
| Initial concentration of species S9b | $3$ | variable |
| Initial concentration of species S10b | $3$ | variable |
| Initial concentration of species S11b | $3$ | variable |
| Initial concentration of species S12b | $3$ | variable |
| Initial concentration of species S13b | $3$ | variable |
| Initial concentration of species S14b | $3$ | variable |
| Initial concentration of species S15b | $3$ | variable |
| Initial concentration of species S16b | $3$ | variable |
| Initial concentration of species S17b | $3$ | variable |
| Initial concentration of species S18b | $3$ | variable |
| Initial concentration of species S19b | $3$ | variable |
| Initial concentration of species S20b | $3$ | variable |
| Initial concentration of species S21b | $3$ | variable |
| Initial concentration of species S22b | $3$ | variable |
| Initial concentration of species S23b | $3$ | variable |
| Initial concentration of species S24b | $3$ | variable |
| Initial concentration of species S25b | $3$ | variable |
| Initial concentration of species S26b | $3$ | variable |
| Initial concentration of species S27b | $3$ | variable |
| Initial concentration of species S28b | $3$ | variable |
| Initial concentration of species S29b | $3$ | variable |
| Initial concentration of species S30b | $3$ | variable |
| Initial concentration of species S31b | $3$ | variable |
| Initial concentration of species S32b | $3$ | variable |
| Initial concentration of species S33b | $3$ | variable |
| Initial concentration of species S34b | $3$ | variable |
| Initial concentration of species S35b | $3$ | variable |
| Initial concentration of species S36b | $3$ | variable |
| Initial concentration of species S37b | $3$ | variable |
| Initial concentration of species S38b | $3$ | variable |
| Initial concentration of species S39b | $3$ | variable |
| Initial concentration of species S40b | $3$ | variable |
| Initial concentration of species S41b | $3$ | variable |
| Initial concentration of species S42b | $3$ | variable |
| Initial concentration of species S43b | $3$ | variable |
| Initial concentration of species S44b | $3$ | variable |
| Initial concentration of species S45b | $3$ | variable |
| Initial concentration of species S46b | $3$ | variable |
| Initial concentration of species S47b | $3$ | variable |
| Initial concentration of species S48b | $3$ | variable |
| Initial concentration of species S49b | $3$ | variable |
| Initial concentration of species S50b | $3$ | variable |
| Initial concentration of species S51b | $3$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
