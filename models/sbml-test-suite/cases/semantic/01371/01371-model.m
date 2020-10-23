(*

category:        Test
synopsis:        A replaced species whose ID is used in an assignment rule.
componentTags:   AssignmentRule, Compartment, Parameter, Species, comp:ModelDefinition, comp:ReplacedElement, comp:Submodel
testTags:        Amount, BoundaryCondition, InitialValueReassigned, NonConstantParameter, comp:SubmodelOutput
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

 In this model, a replaced submodel species ID is used in an assignment rule.

The 'flattened' version of this hierarchical model contains:
* 1 species (J1)
* 1 parameter (A__x)
* 1 compartment (A_C)

There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | A__x | $J1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species J1 | $5$ | variable |
| Initial value of parameter A__x | $J1$ | variable |
| Initial volume of compartment 'A_C' | $1$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
