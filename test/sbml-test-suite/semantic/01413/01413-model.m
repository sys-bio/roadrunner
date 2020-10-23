(*

category:        Test
synopsis:        A delay with a function definition as an argument
componentTags:   AssignmentRule, CSymbolDelay, FunctionDefinition, Parameter, RateRule
testTags:        InitialValueReassigned, NonConstantParameter
testType:        TimeCourse
levels:          2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
generatedBy:     Numeric
packagesPresent: 

 In this model, a delay function takes a function definition as an argument.

The model contains:
* 2 parameters (P1, S1)

There are 2 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | P1 | $delay(S1, addtwo(0.5, 0.5))$ |
| Rate | S1 | $-0.1 * S1$ |]

The model contains the following function definition:

[{width:30em,margin: 1em auto}|  *Id*  |  *Arguments*  |  *Formula*  |
 | addtwo | a, b | $a + b$ |
]
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter P1 | $delay(S1, addtwo(0.5, 0.5))$ | variable |
| Initial value of parameter S1 | $5$ | variable |]

*)
