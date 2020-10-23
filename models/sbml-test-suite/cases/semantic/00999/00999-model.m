Previous version of this file:  
(*

category:      Test
synopsis:      Testing different versions of initialAmount vs. initialConcentration with hasOnlySubstanceUnits=true/false, in a constant and varying compartment.
componentTags: AssignmentRule, Compartment, Parameter, RateRule, Species
testTags:      Amount, HasOnlySubstanceUnits, MultiCompartment, NonConstantCompartment, NonConstantParameter, NonUnityCompartment, InitialValueReassigned
testType:      TimeCourse
levels:        2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:   Analytic

 This model has four species in a constant (non-unity) compartment with the four different combinations of being set by initialAmount or initialConcentration, and being set 'hasOnlySubstanceUnits' equal to true or false.  It also has another four species with the same spread, in a different compartment whose volume changes over time.  Each of these eight species is assigned to a parameter, to catch the 'hasOnlySubstanceUnits' differences (every other parameter has units of amount or concentration, respectively).

The model contains:
* 8 species (S1, S2, S3, S4, S5, S6, S7, S8)
* 8 parameters (p1, p2, p3, p4, p5, p6, p7, p8)
* 2 compartments (comp1, comp2)

There are 9 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | comp2 | $1$ |
| Assignment | p1 | $S1$ |
| Assignment | p2 | $S2$ |
| Assignment | p3 | $S3$ |
| Assignment | p4 | $S4$ |
| Assignment | p5 | $S5$ |
| Assignment | p6 | $S6$ |
| Assignment | p7 | $S7$ |
| Assignment | p8 | $S8$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $1$ | variable |
| Initial concentration of species S2 | $2$ | variable |
| Initial amount of species S3 | $3$ | variable |
| Initial amount of species S4 | $4$ | variable |
| Initial concentration of species S5 | $5$ | variable |
| Initial concentration of species S6 | $6$ | variable |
| Initial amount of species S7 | $7$ | variable |
| Initial amount of species S8 | $8$ | variable |
| Initial value of parameter p1 | $S1$ | variable |
| Initial value of parameter p2 | $S2$ | variable |
| Initial value of parameter p3 | $S3$ | variable |
| Initial value of parameter p4 | $S4$ | variable |
| Initial value of parameter p5 | $S5$ | variable |
| Initial value of parameter p6 | $S6$ | variable |
| Initial value of parameter p7 | $S7$ | variable |
| Initial value of parameter p8 | $S8$ | variable |
| Initial volume of compartment 'comp1' | $10$ | constant |
| Initial volume of compartment 'comp2' | $10$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

