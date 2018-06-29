(*

category:        Test
synopsis:        A chain of assignment rules that cross the submodel boundary
componentTags:   AssignmentRule, Parameter, comp:ModelDefinition, comp:Port, comp:ReplacedElement, comp:Submodel
testTags:        InitialValueReassigned, NonConstantParameter
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

 In this model, a chain of assignment rules assign a to b, b to c, c to d, etc.  However, every other assignment occurs in the parent model, alternating with assignments in the submodel.  This is designed to check simulators that might treat submodels separately from parent models, to make sure they can follow a chain of assignments across the parent/child boundary.

The 'flattened' version of this hierarchical model contains:
* 7 parameters (a, b, c, d, e, f, g)

There are 6 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | c | $b$ |
| Assignment | e | $d$ |
| Assignment | g | $f$ |
| Assignment | b | $a$ |
| Assignment | d | $c$ |
| Assignment | f | $e$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter a | $1$ | constant |
| Initial value of parameter b | $a$ | variable |
| Initial value of parameter c | $b$ | variable |
| Initial value of parameter d | $c$ | variable |
| Initial value of parameter e | $d$ | variable |
| Initial value of parameter f | $e$ | variable |
| Initial value of parameter g | $f$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
