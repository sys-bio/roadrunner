(*

category:        Test
synopsis:        A hierarchical model with three levels of time and extent conversion factors.
componentTags:   CSymbolTime, Compartment, Parameter, Reaction, Species, comp:ModelDefinition, comp:Submodel
testTags:        Amount, HasOnlySubstanceUnits, comp:ExtentConversionFactor, comp:SubmodelOutput, comp:TimeConversionFactor
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

 The reaction in the sub-submodel is affected by time and extent conversion factors three levels deep.

The 'flattened' version of this hierarchical model contains:
* 1 species (sub1__sub1__sub1__s1)
* 10 parameters (timeconv, extentconv, sub1__timeconv, sub1__extentconv, sub1__timeconv_times_timeconv, sub1__extentconv_times_extentconv, sub1__sub1__timeconv, sub1__sub1__extentconv, sub1__sub1__timeconv_times_sub1__timeconv_times_timeconv, sub1__sub1__extentconv_times_sub1__extentconv_times_extentconv)
* 1 compartment (sub1__sub1__sub1__C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| -> sub1__sub1__sub1__s1 | $sub1__sub1__extentconv_times_sub1__extentconv_times_extentconv / sub1__sub1__timeconv_times_sub1__timeconv_times_timeconv * 1000000000 * sub1__sub1__sub1__s1 * (time / sub1__sub1__timeconv_times_sub1__timeconv_times_timeconv)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species sub1__sub1__sub1__s1 | $0.001$ | variable |
| Initial value of parameter timeconv | $60$ | constant |
| Initial value of parameter extentconv | $10$ | constant |
| Initial value of parameter sub1__timeconv | $60$ | constant |
| Initial value of parameter sub1__extentconv | $10$ | constant |
| Initial value of parameter sub1__timeconv_times_timeconv | $sub1__timeconv * timeconv$ | constant |
| Initial value of parameter sub1__extentconv_times_extentconv | $sub1__extentconv * extentconv$ | constant |
| Initial value of parameter sub1__sub1__timeconv | $60$ | constant |
| Initial value of parameter sub1__sub1__extentconv | $10$ | constant |
| Initial value of parameter sub1__sub1__timeconv_times_sub1__timeconv_times_timeconv | $sub1__sub1__timeconv * sub1__timeconv_times_timeconv$ | constant |
| Initial value of parameter sub1__sub1__extentconv_times_sub1__extentconv_times_extentconv | $sub1__sub1__extentconv * sub1__extentconv_times_extentconv$ | constant |
| Initial volume of compartment 'sub1__sub1__sub1__C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

