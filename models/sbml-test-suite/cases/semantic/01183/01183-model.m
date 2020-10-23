(*

category:        Test
 synopsis:        A hierarchical model with time and extent conversion and a reference to a modified replaced reaction, itself with a conversion factor.
componentTags:   AssignmentRule, Compartment, Parameter, Reaction, Species, comp:ModelDefinition, comp:ReplacedElement, comp:Submodel
testTags:        Amount, HasOnlySubstanceUnits, InitialValueReassigned, MultiCompartment, comp:ConversionFactor, comp:ExtentConversionFactor, comp:SubmodelOutput , comp:TimeConversionFactor
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

 In the submodel, there is a reference to a reaction's SId in an assignment rule math.  This reference would normally be modified by the time and extent conversion factors, however, the reaction is replaced, meaning that the only conversion factor to be applied is the conversion factor on the replacement (which, unlike test 1182, exists in this model).  

The 'flattened' version of this hierarchical model contains:
* 2 species (s1, sub1__s1)
* 4 parameters (extentconv, timeconv, extentpertimeconv, sub1__p80)
* 2 compartments (C, sub1__C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| -> s1 | $10$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | sub1__p80 | $J0 / extentpertimeconv + 6$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species s1 | $1$ | variable |
| Initial amount of species sub1__s1 | $1$ | variable |
| Initial value of parameter extentconv | $1000$ | constant |
| Initial value of parameter timeconv | $60$ | constant |
| Initial value of parameter extentpertimeconv | $16.6667$ | constant |
| Initial value of parameter sub1__p80 | $J0 / extentpertimeconv + 6$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |
| Initial volume of compartment 'sub1__C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

