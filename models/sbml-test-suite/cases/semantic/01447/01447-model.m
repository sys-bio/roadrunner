(*

category:        Test
synopsis:        Multiple species references to the same species with assigned variable stoichiometries
componentTags:   CSymbolTime, Compartment, EventNoDelay, InitialAssignment, Reaction, Species
testTags:        Amount, AssignedVariableStoichiometry, InitialValueReassigned, NonUnityStoichiometry
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Numeric
packagesPresent: 

 In this test, a single reaction has multiple products that all point to two species, and each has an assigned value.  This is equivalent to a reaction where all of the species references that point to the same species are collapsed into a single species reference with a stoichiometry equal to the sum of the stoichiometries of the original species references. Partway through the simulation, the stoichiometries change due to an event.

The model contains:
* 2 species (A, B)
* 1 compartment (C)
* 4 species references (A_sr1, A_sr2, B_sr1, B_sr2)

There is one reaction:

[{width:30em,margin: 1em auto}|  *Reaction*  |  *Rate*  |
| J0: A_sr1 A + A_sr2 A -> B_sr1 B + B_sr2 B | $1$ |]
Note:  the following stoichiometries are set separately:  A_sr1, A_sr2, B_sr1, B_sr2


There is one event:

[{width:30em,margin: 1em auto}|  *Event*  |  *Trigger*  | *Event Assignments* |
| _E0 | $time >= 5$ | $A_sr1 = 1$ |
|  |  | $A_sr2 = 3$ |
|  |  | $B_sr1 = 1$ |
|  |  | $B_sr2 = 3$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial concentration of species A | $50$ | variable |
| Initial concentration of species B | $1$ | variable |
| Initial volume of compartment 'C' | $1$ | constant |
| Initial value of species reference 'A_sr1' | $2$ | variable |
| Initial value of species reference 'A_sr2' | $1$ | variable |
| Initial value of species reference 'B_sr1' | $2$ | variable |
| Initial value of species reference 'B_sr2' | $1$ | variable |]

*)
