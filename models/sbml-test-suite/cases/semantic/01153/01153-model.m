(*

category:        Test
synopsis:        A hierarchical model with a deleted rate rule and initial assignment
componentTags:   InitialAssignment, Parameter, comp:Deletion, comp:ModelDefinition, comp:Port, comp:ReplacedElement, comp:SBaseRef, comp:Submodel
testTags:        InitialValueReassigned, comp:SubmodelOutput
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

 The parent model in this document contains deletions that remove an initial assignment and a rate rule two levels down.  The parent model contains only an initial assignment for that parameter, leaving it constant.

The 'flattened' version of this hierarchical model contains:
* 4 parameters (z, i, A__A__y, A__A__Q)
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter i | $2$ | constant |
| Initial value of parameter A__A__y | $7$ | constant |
| Initial value of parameter A__A__Q | $2$ | constant |
| Initial value of parameter z | $34 * i$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

