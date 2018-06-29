(*

category:        Test
synopsis:        A larger model where the order of assignment rules is important.
componentTags:   AssignmentRule, Compartment, Parameter, Reaction, Species
testTags:        Amount, InitialValueReassigned, NonConstantParameter
testType:        TimeCourse
levels:          2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Numeric
packagesPresent: 

 In this real model of a reaction system, the order of the assignment rules makes a difference in how the simulation proceeds: it is important to carry out the assignments such that the independent assignments are performed first, with the dependent assignments carried out after the assignments to variables they each depend on.  This has been seen to make an appreciable difference when using large timesteps and the Euler method.

The model contains:
* 18 species (pp9_mrna, p9, pp8_mrna, p8, pp7_mrna, p7, pp6_mrna, p6, pp5_mrna, p5, pp4_mrna, p4, pp3_mrna, p3, pp2_mrna, p2, pp1_mrna, p1)
* 90 parameters (pp8_mrna_degradation_rate, pp9_mrna_degradation_rate, p1_degradation_rate, p2_degradation_rate, p3_degradation_rate, p4_degradation_rate, p5_degradation_rate, p6_degradation_rate, p7_degradation_rate, p8_degradation_rate, p9_degradation_rate, v1_Kd, v1_h, v2_Kd, v2_h, v3_Kd, v3_h, v4_Kd, v4_h, v5_Kd, v5_h, v6_Kd, v6_h, v7_Kd, v7_h, v8_Kd, v8_h, v9_Kd, v9_h, pp1_mrna_degradation_rate, pp2_mrna_degradation_rate, pro1_strength, pro2_strength, pro3_strength, pro4_strength, pro5_strength, pro6_strength, pro7_strength, pro8_strength, pro9_strength, pp3_mrna_degradation_rate, v10_Kd, v10_h, v11_Kd, v11_h, v12_Kd, v12_h, v13_Kd, v13_h, pp4_mrna_degradation_rate, v14_Kd, v14_h, v15_Kd, v15_h, pp5_mrna_degradation_rate, rbs1_strength, rbs2_strength, rbs3_strength, rbs4_strength, rbs5_strength, rbs6_strength, rbs7_strength, rbs8_strength, rbs9_strength, pp6_mrna_degradation_rate, pp7_mrna_degradation_rate, as8, cod1, cod2, cod3, cod4, cod5, cod6, cod7, cod8, cod9, rs1, rs2, rs3, rs4, rs5, rs6, rs7, as1, as2, as3, as4, as5, as6, as7)
* 1 compartment (DefaultCompartment)

There are 36 reactions:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| pp9_v1: -> pp9_mrna | $cod9$ |
| pp9_v2: pp9_mrna -> | $pp9_mrna_degradation_rate * pp9_mrna$ |
| pp9_v3: -> p9 | $rbs9_strength * pp9_mrna$ |
| pp9_v4: p9 -> | $p9_degradation_rate * p9$ |
| pp8_v1: -> pp8_mrna | $cod8$ |
| pp8_v2: pp8_mrna -> | $pp8_mrna_degradation_rate * pp8_mrna$ |
| pp8_v3: -> p8 | $rbs8_strength * pp8_mrna$ |
| pp8_v4: p8 -> | $p8_degradation_rate * p8$ |
| pp7_v1: -> pp7_mrna | $cod7$ |
| pp7_v2: pp7_mrna -> | $pp7_mrna_degradation_rate * pp7_mrna$ |
| pp7_v3: -> p7 | $rbs7_strength * pp7_mrna$ |
| pp7_v4: p7 -> | $p7_degradation_rate * p7$ |
| pp6_v1: -> pp6_mrna | $cod6$ |
| pp6_v2: pp6_mrna -> | $pp6_mrna_degradation_rate * pp6_mrna$ |
| pp6_v3: -> p6 | $rbs6_strength * pp6_mrna$ |
| pp6_v4: p6 -> | $p6_degradation_rate * p6$ |
| pp5_v1: -> pp5_mrna | $cod5$ |
| pp5_v2: pp5_mrna -> | $pp5_mrna_degradation_rate * pp5_mrna$ |
| pp5_v3: -> p5 | $rbs5_strength * pp5_mrna$ |
| pp5_v4: p5 -> | $p5_degradation_rate * p5$ |
| pp4_v1: -> pp4_mrna | $cod4$ |
| pp4_v2: pp4_mrna -> | $pp4_mrna_degradation_rate * pp4_mrna$ |
| pp4_v3: -> p4 | $rbs4_strength * pp4_mrna$ |
| pp4_v4: p4 -> | $p4_degradation_rate * p4$ |
| pp3_v1: -> pp3_mrna | $cod3$ |
| pp3_v2: pp3_mrna -> | $pp3_mrna_degradation_rate * pp3_mrna$ |
| pp3_v3: -> p3 | $rbs3_strength * pp3_mrna$ |
| pp3_v4: p3 -> | $p3_degradation_rate * p3$ |
| pp2_v1: -> pp2_mrna | $cod2$ |
| pp2_v2: pp2_mrna -> | $pp2_mrna_degradation_rate * pp2_mrna$ |
| pp2_v3: -> p2 | $rbs2_strength * pp2_mrna$ |
| pp2_v4: p2 -> | $p2_degradation_rate * p2$ |
| pp1_v1: -> pp1_mrna | $cod1$ |
| pp1_v2: pp1_mrna -> | $pp1_mrna_degradation_rate * pp1_mrna$ |
| pp1_v3: -> p1 | $rbs1_strength * pp1_mrna$ |
| pp1_v4: p1 -> | $p1_degradation_rate * p1$ |]


There are 24 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | as8 | $(1 + (p5 / v9_Kd)^v9_h - 1) / (1 + (p5 / v9_Kd)^v9_h)$ |
| Assignment | cod1 | $pro1_strength$ |
| Assignment | cod2 | $pro2_strength * as1 * rs1$ |
| Assignment | cod3 | $pro3_strength * rs2 * rs3$ |
| Assignment | cod4 | $pro4_strength * rs7 * rs6$ |
| Assignment | cod5 | $pro5_strength * as2$ |
| Assignment | cod6 | $pro6_strength * (as3 + as4)$ |
| Assignment | cod7 | $pro7_strength * (as7 + as8)$ |
| Assignment | cod8 | $pro8_strength * as5 * rs4$ |
| Assignment | cod9 | $pro9_strength * as6 * rs5$ |
| Assignment | rs1 | $1 / (1 + (p9 / v13_Kd)^v13_h)$ |
| Assignment | rs2 | $1 / (1 + (p2 / v2_Kd)^v2_h)$ |
| Assignment | rs3 | $1 / (1 + (p3 / v3_Kd)^v3_h)$ |
| Assignment | rs4 | $1 / (1 + (p8 / v11_Kd)^v11_h)$ |
| Assignment | rs5 | $1 / (1 + (p8 / v12_Kd)^v12_h)$ |
| Assignment | rs6 | $1 / (1 + (p2 / v14_Kd)^v14_h)$ |
| Assignment | rs7 | $1 / (1 + (p3 / v15_Kd)^v15_h)$ |
| Assignment | as1 | $(1 + (p1 / v1_Kd)^v1_h - 1) / (1 + (p1 / v1_Kd)^v1_h)$ |
| Assignment | as2 | $(1 + (p4 / v4_Kd)^v4_h - 1) / (1 + (p4 / v4_Kd)^v4_h)$ |
| Assignment | as3 | $(1 + (p5 / v5_Kd)^v5_h - 1) / (1 + (p5 / v5_Kd)^v5_h)$ |
| Assignment | as4 | $(1 + (p6 / v6_Kd)^v6_h - 1) / (1 + (p6 / v6_Kd)^v6_h)$ |
| Assignment | as5 | $(1 + (p7 / v7_Kd)^v7_h - 1) / (1 + (p7 / v7_Kd)^v7_h)$ |
| Assignment | as6 | $(1 + (p7 / v10_Kd)^v10_h - 1) / (1 + (p7 / v10_Kd)^v10_h)$ |
| Assignment | as7 | $(1 + (p6 / v8_Kd)^v8_h - 1) / (1 + (p6 / v8_Kd)^v8_h)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species pp9_mrna | $0$ | variable |
| Initial amount of species p9 | $0$ | variable |
| Initial amount of species pp8_mrna | $0$ | variable |
| Initial amount of species p8 | $0$ | variable |
| Initial amount of species pp7_mrna | $0$ | variable |
| Initial amount of species p7 | $0$ | variable |
| Initial amount of species pp6_mrna | $0$ | variable |
| Initial amount of species p6 | $0$ | variable |
| Initial amount of species pp5_mrna | $0$ | variable |
| Initial amount of species p5 | $0$ | variable |
| Initial amount of species pp4_mrna | $0$ | variable |
| Initial amount of species p4 | $0$ | variable |
| Initial amount of species pp3_mrna | $0$ | variable |
| Initial amount of species p3 | $0$ | variable |
| Initial amount of species pp2_mrna | $0$ | variable |
| Initial amount of species p2 | $0$ | variable |
| Initial amount of species pp1_mrna | $0$ | variable |
| Initial amount of species p1 | $5$ | variable |
| Initial value of parameter pp8_mrna_degradation_rate | $1$ | constant |
| Initial value of parameter pp9_mrna_degradation_rate | $1$ | constant |
| Initial value of parameter p1_degradation_rate | $0.5$ | constant |
| Initial value of parameter p2_degradation_rate | $0.5$ | constant |
| Initial value of parameter p3_degradation_rate | $0.5$ | constant |
| Initial value of parameter p4_degradation_rate | $0.5$ | constant |
| Initial value of parameter p5_degradation_rate | $0.5$ | constant |
| Initial value of parameter p6_degradation_rate | $0.5$ | constant |
| Initial value of parameter p7_degradation_rate | $0.5$ | constant |
| Initial value of parameter p8_degradation_rate | $0.5$ | constant |
| Initial value of parameter p9_degradation_rate | $0.5$ | constant |
| Initial value of parameter v1_Kd | $11.147$ | constant |
| Initial value of parameter v1_h | $1$ | constant |
| Initial value of parameter v2_Kd | $1$ | constant |
| Initial value of parameter v2_h | $4$ | constant |
| Initial value of parameter v3_Kd | $20$ | constant |
| Initial value of parameter v3_h | $1$ | constant |
| Initial value of parameter v4_Kd | $0.2$ | constant |
| Initial value of parameter v4_h | $4$ | constant |
| Initial value of parameter v5_Kd | $0.2$ | constant |
| Initial value of parameter v5_h | $4$ | constant |
| Initial value of parameter v6_Kd | $0.04$ | constant |
| Initial value of parameter v6_h | $4$ | constant |
| Initial value of parameter v7_Kd | $0.02$ | constant |
| Initial value of parameter v7_h | $4$ | constant |
| Initial value of parameter v8_Kd | $0.04$ | constant |
| Initial value of parameter v8_h | $4$ | constant |
| Initial value of parameter v9_Kd | $0.2$ | constant |
| Initial value of parameter v9_h | $4$ | constant |
| Initial value of parameter pp1_mrna_degradation_rate | $1$ | constant |
| Initial value of parameter pp2_mrna_degradation_rate | $1$ | constant |
| Initial value of parameter pro1_strength | $2$ | constant |
| Initial value of parameter pro2_strength | $4.5077$ | constant |
| Initial value of parameter pro3_strength | $5$ | constant |
| Initial value of parameter pro4_strength | $5$ | constant |
| Initial value of parameter pro5_strength | $5$ | constant |
| Initial value of parameter pro6_strength | $1.31$ | constant |
| Initial value of parameter pro7_strength | $1.31$ | constant |
| Initial value of parameter pro8_strength | $5$ | constant |
| Initial value of parameter pro9_strength | $5$ | constant |
| Initial value of parameter pp3_mrna_degradation_rate | $1$ | constant |
| Initial value of parameter v10_Kd | $0.02$ | constant |
| Initial value of parameter v10_h | $4$ | constant |
| Initial value of parameter v11_Kd | $0.1$ | constant |
| Initial value of parameter v11_h | $2$ | constant |
| Initial value of parameter v12_Kd | $0.1$ | constant |
| Initial value of parameter v12_h | $2$ | constant |
| Initial value of parameter v13_Kd | $0.01$ | constant |
| Initial value of parameter v13_h | $2$ | constant |
| Initial value of parameter pp4_mrna_degradation_rate | $1$ | constant |
| Initial value of parameter v14_Kd | $1$ | constant |
| Initial value of parameter v14_h | $4$ | constant |
| Initial value of parameter v15_Kd | $20$ | constant |
| Initial value of parameter v15_h | $1$ | constant |
| Initial value of parameter pp5_mrna_degradation_rate | $1$ | constant |
| Initial value of parameter rbs1_strength | $0.3668$ | constant |
| Initial value of parameter rbs2_strength | $1.4102$ | constant |
| Initial value of parameter rbs3_strength | $0.8$ | constant |
| Initial value of parameter rbs4_strength | $2.21$ | constant |
| Initial value of parameter rbs5_strength | $0.5$ | constant |
| Initial value of parameter rbs6_strength | $2$ | constant |
| Initial value of parameter rbs7_strength | $5$ | constant |
| Initial value of parameter rbs8_strength | $3.6377$ | constant |
| Initial value of parameter rbs9_strength | $8$ | constant |
| Initial value of parameter pp6_mrna_degradation_rate | $1$ | constant |
| Initial value of parameter pp7_mrna_degradation_rate | $1$ | constant |
| Initial value of parameter as8 | $(1 + (p5 / v9_Kd)^v9_h - 1) / (1 + (p5 / v9_Kd)^v9_h)$ | variable |
| Initial value of parameter cod1 | $pro1_strength$ | variable |
| Initial value of parameter cod2 | $pro2_strength * as1 * rs1$ | variable |
| Initial value of parameter cod3 | $pro3_strength * rs2 * rs3$ | variable |
| Initial value of parameter cod4 | $pro4_strength * rs7 * rs6$ | variable |
| Initial value of parameter cod5 | $pro5_strength * as2$ | variable |
| Initial value of parameter cod6 | $pro6_strength * (as3 + as4)$ | variable |
| Initial value of parameter cod7 | $pro7_strength * (as7 + as8)$ | variable |
| Initial value of parameter cod8 | $pro8_strength * as5 * rs4$ | variable |
| Initial value of parameter cod9 | $pro9_strength * as6 * rs5$ | variable |
| Initial value of parameter rs1 | $1 / (1 + (p9 / v13_Kd)^v13_h)$ | variable |
| Initial value of parameter rs2 | $1 / (1 + (p2 / v2_Kd)^v2_h)$ | variable |
| Initial value of parameter rs3 | $1 / (1 + (p3 / v3_Kd)^v3_h)$ | variable |
| Initial value of parameter rs4 | $1 / (1 + (p8 / v11_Kd)^v11_h)$ | variable |
| Initial value of parameter rs5 | $1 / (1 + (p8 / v12_Kd)^v12_h)$ | variable |
| Initial value of parameter rs6 | $1 / (1 + (p2 / v14_Kd)^v14_h)$ | variable |
| Initial value of parameter rs7 | $1 / (1 + (p3 / v15_Kd)^v15_h)$ | variable |
| Initial value of parameter as1 | $(1 + (p1 / v1_Kd)^v1_h - 1) / (1 + (p1 / v1_Kd)^v1_h)$ | variable |
| Initial value of parameter as2 | $(1 + (p4 / v4_Kd)^v4_h - 1) / (1 + (p4 / v4_Kd)^v4_h)$ | variable |
| Initial value of parameter as3 | $(1 + (p5 / v5_Kd)^v5_h - 1) / (1 + (p5 / v5_Kd)^v5_h)$ | variable |
| Initial value of parameter as4 | $(1 + (p6 / v6_Kd)^v6_h - 1) / (1 + (p6 / v6_Kd)^v6_h)$ | variable |
| Initial value of parameter as5 | $(1 + (p7 / v7_Kd)^v7_h - 1) / (1 + (p7 / v7_Kd)^v7_h)$ | variable |
| Initial value of parameter as6 | $(1 + (p7 / v10_Kd)^v10_h - 1) / (1 + (p7 / v10_Kd)^v10_h)$ | variable |
| Initial value of parameter as7 | $(1 + (p6 / v8_Kd)^v8_h - 1) / (1 + (p6 / v8_Kd)^v8_h)$ | variable |
| Initial volume of compartment 'DefaultCompartment' | $1$ | constant |]

The species' initial quantities are given in terms of substance units to
make it easier to use the model in a discrete stochastic simulator, but
their symbols represent their values in concentration units where they
appear in expressions.

*)
