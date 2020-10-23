(*

category:        Test
synopsis:        A hierarchical model with extent conversion and a reference to a modified reaction.
componentTags:   AssignmentRule, Compartment, Parameter, Reaction, Species, comp:ModelDefinition, comp:Submodel
testTags:        Amount, HasOnlySubstanceUnits, InitialValueReassigned, comp:ExtentConversionFactor, comp:SubmodelOutput
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: comp

 In the submodel, there is a reference to a reaction's SId in an assignment rule math.  This reference must be modified by the extent conversion factor just as the kinetic law itself is modified by the extent conversion factor.

The 'flattened' version of this hierarchical model contains:
* 1 species (sub1__s1)
* 2 parameters (extentconv, sub1__p80)
* 1 compartment (sub1__C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| -> sub1__s1 | $extentconv * 10$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | sub1__p80 | $sub1__J0 / extentconv + 6$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species sub1__s1 | $1$ | variable |
| Initial value of parameter extentconv | $1000$ | constant |
| Initial value of parameter sub1__p80 | $sub1__J0 / extentconv + 6$ | variable |
| Initial volume of compartment 'sub1__C' | $1$ | constant |]

*)

