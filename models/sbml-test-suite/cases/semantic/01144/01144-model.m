(*

category:        Test
synopsis:        A hierarchical model with a submodel reaction modified by time and extent conversion factors.
componentTags:   CSymbolTime, Compartment, Parameter, Reaction, Species, comp:ModelDefinition, comp:ReplacedElement, comp:Submodel
testTags:        Amount, HasOnlySubstanceUnits, comp:ExtentConversionFactor, comp:SubmodelOutput, comp:TimeConversionFactor
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

The kinetic law of the reaction in the submodel in this model must be modified by both the time conversion factor and the extent conversion factor of the parent model.

The 'flattened' version of this hierarchical model contains:
* 1 species (sub1__s1)
* 3 parameters (timeconv, extentconv, t3)
* 1 compartment (sub1__C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| -> sub1__s1 | $extentconv / timeconv * sub1__s1 * t3 * (time / timeconv)$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species sub1__s1 | $1$ | variable |
| Initial value of parameter timeconv | $60$ | constant |
| Initial value of parameter extentconv | $1000$ | constant |
| Initial value of parameter t3 | $0.2$ | constant |
| Initial volume of compartment 'sub1__C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

