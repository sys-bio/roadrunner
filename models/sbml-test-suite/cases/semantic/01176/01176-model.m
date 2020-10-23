(*

category:        Test
synopsis:        A hierarchical model with a time conversion factor and a delay in a kinetic law.
componentTags:   AssignmentRule, CSymbolDelay, CSymbolTime, Compartment, Parameter, Reaction, Species, comp:ModelDefinition, comp:ReplacedBy, comp:ReplacedElement, comp:Submodel
testTags:        Amount, HasOnlySubstanceUnits, InitialValueReassigned, NonConstantParameter, comp:TimeConversionFactor
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

This model contains a submodel with a reaction whose kinetic law contains a delay, contained in a model with a time conversion factor.

The 'flattened' version of this hierarchical model contains:
* 1 species (s1)
* 2 parameters (timeconv, t5)
* 1 compartment (C)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| -> s1 | $1 / timeconv * delay(t5, timeconv * 0.2) * (time / timeconv)$ |]


There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | t5 | $time / timeconv$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial amount of species s1 | $0.001$ | variable |
| Initial value of parameter timeconv | $60$ | constant |
| Initial value of parameter t5 | $time / timeconv$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

