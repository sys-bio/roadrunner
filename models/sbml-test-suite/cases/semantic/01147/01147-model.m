(*

category:        Test
synopsis:        A hierarchical model with two levels of time conversion factors.
componentTags:   AssignmentRule, CSymbolTime, InitialAssignment, Parameter, comp:ModelDefinition, comp:Submodel
testTags:        InitialValueReassigned, NonConstantParameter, comp:SubmodelOutput, comp:TimeConversionFactor
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: comp

 The assignment rule in the sub-submodel must be converted by time conversion factors at both levels of submodel creation.

The 'flattened' version of this hierarchical model contains:
* 4 parameters (timeconv, sub1__timeconv, sub1__timeconv_times_timeconv, sub1__sub1__t1)

There is one rule:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | sub1__sub1__t1 | $time / sub1__timeconv_times_timeconv + 3$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter timeconv | $60$ | constant |
| Initial value of parameter sub1__timeconv | $60$ | constant |
| Initial value of parameter sub1__timeconv_times_timeconv | $sub1__timeconv * timeconv$ | constant |
| Initial value of parameter sub1__sub1__t1 | $time / sub1__timeconv_times_timeconv + 3$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

