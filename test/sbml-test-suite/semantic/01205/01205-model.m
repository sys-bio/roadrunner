(*

category:        Test
synopsis:        A species concentration under control of a rate rule.
componentTags:   AssignmentRule, Compartment, Parameter, RateRule, Species
testTags:        Amount, BoundaryCondition, InitialValueReassigned, NonConstantParameter, NonUnityCompartment
testType:        TimeCourse
levels:          1.2, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Numeric
packagesPresent: 

 A species is under control of a rate rule, and set 'hasOnlySubstanceUnits=false', meaning that the species concentration is affected by this rate of change, but the volume of the compartment stays constant.  That value then gets assigned to the parameter 'x', since we typically only require Species output in amounts.

The model contains:
* 1 species (S1)
* 1 parameter (x)
* 1 compartment (C1)

There are 2 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Rate | S1 | $0.4$ |
| Assignment | x | $S1$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species S1 | $0$ | variable |
| Initial value of parameter x | $S1$ | variable |
| Initial volume of compartment 'C1' | $0.5$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)

