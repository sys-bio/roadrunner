(*

category:        Test
synopsis:        A hierarchical model with extent and time conversion and a reference to a modified reaction.
componentTags:   AssignmentRule, Compartment, Parameter, Reaction, Species, comp:ModelDefinition, comp:Submodel
testTags:        Amount, HasOnlySubstanceUnits, InitialValueReassigned, comp:ExtentConversionFactor, comp:SubmodelOutput, comp:TimeConversionFactor
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

 In the submodel, there is a reference to a reaction's SId in an assignment rule math.  This reference must be modified by the time and extent conversion factor just as the kinetic law itself is modified by the time and extent conversion factor.

The 'flattened' version of this hierarchical model contains:
* 1 species (sub1__s1)
* 3 parameters (extentconv, timeconv, sub1__p80)
* 1 compartment (sub1__C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| -> sub1__s1 | $extentconv / timeconv * 10$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | sub1__p80 | $sub1__J0 / (extentconv / timeconv) + 6$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species sub1__s1 | $1$ | variable |
| Initial value of parameter extentconv | $1000$ | constant |
| Initial value of parameter timeconv | $60$ | constant |
| Initial value of parameter sub1__p80 | $sub1__J0 / (extentconv / timeconv) + 6$ | variable |
| Initial volume of compartment 'sub1__C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

