(*

category:        Test
synopsis:        A test of random event execution at t0.
componentTags:   AssignmentRule, EventNoDelay, EventPriority, Parameter
testTags:        EventIsNotPersistent, EventT0Firing, InitialValueReassigned, NonConstantParameter, RandomEventExecution
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

 In this model, 200 events 'compete' in pairs to assing values to variables:  if E0 fires, Eb0 will not, and visa versa, for all events E0-E99, Eb0-Eb99.  Because all events have the same priority, the E# set should win about half the time (assigning a value to the corresponding 'A' variable), and the Eb# set should win the other half, assigning values to the corresponding 'B' variables.

 'toodiverged' will only equal 1 if (Asum-Bsum)>100.  Since the standard deviation of 100 coin flips is 5, which would correspond to a difference of 20, 'toodiverged' will only equal 1 if the observed distribution of choices is more than five standard deviations away from expected (0), which will happen approximately one time in 1.7 million.

The model contains:
* 205 parameters (trig, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27, A28, A29, A30, A31, A32, A33, A34, A35, A36, A37, A38, A39, A40, A41, A42, A43, A44, A45, A46, A47, A48, A49, A50, A51, A52, A53, A54, A55, A56, A57, A58, A59, A60, A61, A62, A63, A64, A65, A66, A67, A68, A69, A70, A71, A72, A73, A74, A75, A76, A77, A78, A79, A80, A81, A82, A83, A84, A85, A86, A87, A88, A89, A90, A91, A92, A93, A94, A95, A96, A97, A98, A99, B0, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20, B21, B22, B23, B24, B25, B26, B27, B28, B29, B30, B31, B32, B33, B34, B35, B36, B37, B38, B39, B40, B41, B42, B43, B44, B45, B46, B47, B48, B49, B50, B51, B52, B53, B54, B55, B56, B57, B58, B59, B60, B61, B62, B63, B64, B65, B66, B67, B68, B69, B70, B71, B72, B73, B74, B75, B76, B77, B78, B79, B80, B81, B82, B83, B84, B85, B86, B87, B88, B89, B90, B91, B92, B93, B94, B95, B96, B97, B98, B99, Asum, Bsum, Allsum, toodiverged)

There are 201 events:

[{width:45em,margin: 1em auto}|  *Event*  |  *Trigger*  |  *Priority*  |  *Persistent*  |  *initialValue*  | *Event Assignments* |
| E0 | $(trig > 0) && (A0 < 1)$ | $5$ | false | false | $B0 = 2$ |
| E1 | $(trig > 0) && (A1 < 1)$ | $5$ | false | false | $B1 = 2$ |
| E2 | $(trig > 0) && (A2 < 1)$ | $5$ | false | false | $B2 = 2$ |
| E3 | $(trig > 0) && (A3 < 1)$ | $5$ | false | false | $B3 = 2$ |
| E4 | $(trig > 0) && (A4 < 1)$ | $5$ | false | false | $B4 = 2$ |
| E5 | $(trig > 0) && (A5 < 1)$ | $5$ | false | false | $B5 = 2$ |
| E6 | $(trig > 0) && (A6 < 1)$ | $5$ | false | false | $B6 = 2$ |
| E7 | $(trig > 0) && (A7 < 1)$ | $5$ | false | false | $B7 = 2$ |
| E8 | $(trig > 0) && (A8 < 1)$ | $5$ | false | false | $B8 = 2$ |
| E9 | $(trig > 0) && (A9 < 1)$ | $5$ | false | false | $B9 = 2$ |
| E10 | $(trig > 0) && (A10 < 1)$ | $5$ | false | false | $B10 = 2$ |
| E11 | $(trig > 0) && (A11 < 1)$ | $5$ | false | false | $B11 = 2$ |
| E12 | $(trig > 0) && (A12 < 1)$ | $5$ | false | false | $B12 = 2$ |
| E13 | $(trig > 0) && (A13 < 1)$ | $5$ | false | false | $B13 = 2$ |
| E14 | $(trig > 0) && (A14 < 1)$ | $5$ | false | false | $B14 = 2$ |
| E15 | $(trig > 0) && (A15 < 1)$ | $5$ | false | false | $B15 = 2$ |
| E16 | $(trig > 0) && (A16 < 1)$ | $5$ | false | false | $B16 = 2$ |
| E17 | $(trig > 0) && (A17 < 1)$ | $5$ | false | false | $B17 = 2$ |
| E18 | $(trig > 0) && (A18 < 1)$ | $5$ | false | false | $B18 = 2$ |
| E19 | $(trig > 0) && (A19 < 1)$ | $5$ | false | false | $B19 = 2$ |
| E20 | $(trig > 0) && (A20 < 1)$ | $5$ | false | false | $B20 = 2$ |
| E21 | $(trig > 0) && (A21 < 1)$ | $5$ | false | false | $B21 = 2$ |
| E22 | $(trig > 0) && (A22 < 1)$ | $5$ | false | false | $B22 = 2$ |
| E23 | $(trig > 0) && (A23 < 1)$ | $5$ | false | false | $B23 = 2$ |
| E24 | $(trig > 0) && (A24 < 1)$ | $5$ | false | false | $B24 = 2$ |
| E25 | $(trig > 0) && (A25 < 1)$ | $5$ | false | false | $B25 = 2$ |
| E26 | $(trig > 0) && (A26 < 1)$ | $5$ | false | false | $B26 = 2$ |
| E27 | $(trig > 0) && (A27 < 1)$ | $5$ | false | false | $B27 = 2$ |
| E28 | $(trig > 0) && (A28 < 1)$ | $5$ | false | false | $B28 = 2$ |
| E29 | $(trig > 0) && (A29 < 1)$ | $5$ | false | false | $B29 = 2$ |
| E30 | $(trig > 0) && (A30 < 1)$ | $5$ | false | false | $B30 = 2$ |
| E31 | $(trig > 0) && (A31 < 1)$ | $5$ | false | false | $B31 = 2$ |
| E32 | $(trig > 0) && (A32 < 1)$ | $5$ | false | false | $B32 = 2$ |
| E33 | $(trig > 0) && (A33 < 1)$ | $5$ | false | false | $B33 = 2$ |
| E34 | $(trig > 0) && (A34 < 1)$ | $5$ | false | false | $B34 = 2$ |
| E35 | $(trig > 0) && (A35 < 1)$ | $5$ | false | false | $B35 = 2$ |
| E36 | $(trig > 0) && (A36 < 1)$ | $5$ | false | false | $B36 = 2$ |
| E37 | $(trig > 0) && (A37 < 1)$ | $5$ | false | false | $B37 = 2$ |
| E38 | $(trig > 0) && (A38 < 1)$ | $5$ | false | false | $B38 = 2$ |
| E39 | $(trig > 0) && (A39 < 1)$ | $5$ | false | false | $B39 = 2$ |
| E40 | $(trig > 0) && (A40 < 1)$ | $5$ | false | false | $B40 = 2$ |
| E41 | $(trig > 0) && (A41 < 1)$ | $5$ | false | false | $B41 = 2$ |
| E42 | $(trig > 0) && (A42 < 1)$ | $5$ | false | false | $B42 = 2$ |
| E43 | $(trig > 0) && (A43 < 1)$ | $5$ | false | false | $B43 = 2$ |
| E44 | $(trig > 0) && (A44 < 1)$ | $5$ | false | false | $B44 = 2$ |
| E45 | $(trig > 0) && (A45 < 1)$ | $5$ | false | false | $B45 = 2$ |
| E46 | $(trig > 0) && (A46 < 1)$ | $5$ | false | false | $B46 = 2$ |
| E47 | $(trig > 0) && (A47 < 1)$ | $5$ | false | false | $B47 = 2$ |
| E48 | $(trig > 0) && (A48 < 1)$ | $5$ | false | false | $B48 = 2$ |
| E49 | $(trig > 0) && (A49 < 1)$ | $5$ | false | false | $B49 = 2$ |
| E50 | $(trig > 0) && (A50 < 1)$ | $5$ | false | false | $B50 = 2$ |
| E51 | $(trig > 0) && (A51 < 1)$ | $5$ | false | false | $B51 = 2$ |
| E52 | $(trig > 0) && (A52 < 1)$ | $5$ | false | false | $B52 = 2$ |
| E53 | $(trig > 0) && (A53 < 1)$ | $5$ | false | false | $B53 = 2$ |
| E54 | $(trig > 0) && (A54 < 1)$ | $5$ | false | false | $B54 = 2$ |
| E55 | $(trig > 0) && (A55 < 1)$ | $5$ | false | false | $B55 = 2$ |
| E56 | $(trig > 0) && (A56 < 1)$ | $5$ | false | false | $B56 = 2$ |
| E57 | $(trig > 0) && (A57 < 1)$ | $5$ | false | false | $B57 = 2$ |
| E58 | $(trig > 0) && (A58 < 1)$ | $5$ | false | false | $B58 = 2$ |
| E59 | $(trig > 0) && (A59 < 1)$ | $5$ | false | false | $B59 = 2$ |
| E60 | $(trig > 0) && (A60 < 1)$ | $5$ | false | false | $B60 = 2$ |
| E61 | $(trig > 0) && (A61 < 1)$ | $5$ | false | false | $B61 = 2$ |
| E62 | $(trig > 0) && (A62 < 1)$ | $5$ | false | false | $B62 = 2$ |
| E63 | $(trig > 0) && (A63 < 1)$ | $5$ | false | false | $B63 = 2$ |
| E64 | $(trig > 0) && (A64 < 1)$ | $5$ | false | false | $B64 = 2$ |
| E65 | $(trig > 0) && (A65 < 1)$ | $5$ | false | false | $B65 = 2$ |
| E66 | $(trig > 0) && (A66 < 1)$ | $5$ | false | false | $B66 = 2$ |
| E67 | $(trig > 0) && (A67 < 1)$ | $5$ | false | false | $B67 = 2$ |
| E68 | $(trig > 0) && (A68 < 1)$ | $5$ | false | false | $B68 = 2$ |
| E69 | $(trig > 0) && (A69 < 1)$ | $5$ | false | false | $B69 = 2$ |
| E70 | $(trig > 0) && (A70 < 1)$ | $5$ | false | false | $B70 = 2$ |
| E71 | $(trig > 0) && (A71 < 1)$ | $5$ | false | false | $B71 = 2$ |
| E72 | $(trig > 0) && (A72 < 1)$ | $5$ | false | false | $B72 = 2$ |
| E73 | $(trig > 0) && (A73 < 1)$ | $5$ | false | false | $B73 = 2$ |
| E74 | $(trig > 0) && (A74 < 1)$ | $5$ | false | false | $B74 = 2$ |
| E75 | $(trig > 0) && (A75 < 1)$ | $5$ | false | false | $B75 = 2$ |
| E76 | $(trig > 0) && (A76 < 1)$ | $5$ | false | false | $B76 = 2$ |
| E77 | $(trig > 0) && (A77 < 1)$ | $5$ | false | false | $B77 = 2$ |
| E78 | $(trig > 0) && (A78 < 1)$ | $5$ | false | false | $B78 = 2$ |
| E79 | $(trig > 0) && (A79 < 1)$ | $5$ | false | false | $B79 = 2$ |
| E80 | $(trig > 0) && (A80 < 1)$ | $5$ | false | false | $B80 = 2$ |
| E81 | $(trig > 0) && (A81 < 1)$ | $5$ | false | false | $B81 = 2$ |
| E82 | $(trig > 0) && (A82 < 1)$ | $5$ | false | false | $B82 = 2$ |
| E83 | $(trig > 0) && (A83 < 1)$ | $5$ | false | false | $B83 = 2$ |
| E84 | $(trig > 0) && (A84 < 1)$ | $5$ | false | false | $B84 = 2$ |
| E85 | $(trig > 0) && (A85 < 1)$ | $5$ | false | false | $B85 = 2$ |
| E86 | $(trig > 0) && (A86 < 1)$ | $5$ | false | false | $B86 = 2$ |
| E87 | $(trig > 0) && (A87 < 1)$ | $5$ | false | false | $B87 = 2$ |
| E88 | $(trig > 0) && (A88 < 1)$ | $5$ | false | false | $B88 = 2$ |
| E89 | $(trig > 0) && (A89 < 1)$ | $5$ | false | false | $B89 = 2$ |
| E90 | $(trig > 0) && (A90 < 1)$ | $5$ | false | false | $B90 = 2$ |
| E91 | $(trig > 0) && (A91 < 1)$ | $5$ | false | false | $B91 = 2$ |
| E92 | $(trig > 0) && (A92 < 1)$ | $5$ | false | false | $B92 = 2$ |
| E93 | $(trig > 0) && (A93 < 1)$ | $5$ | false | false | $B93 = 2$ |
| E94 | $(trig > 0) && (A94 < 1)$ | $5$ | false | false | $B94 = 2$ |
| E95 | $(trig > 0) && (A95 < 1)$ | $5$ | false | false | $B95 = 2$ |
| E96 | $(trig > 0) && (A96 < 1)$ | $5$ | false | false | $B96 = 2$ |
| E97 | $(trig > 0) && (A97 < 1)$ | $5$ | false | false | $B97 = 2$ |
| E98 | $(trig > 0) && (A98 < 1)$ | $5$ | false | false | $B98 = 2$ |
| E99 | $(trig > 0) && (A99 < 1)$ | $5$ | false | false | $B99 = 2$ |
| Eb0 | $(trig > 0) && (B0 < 1)$ | $5$ | false | false | $A0 = 2$ |
| Eb1 | $(trig > 0) && (B1 < 1)$ | $5$ | false | false | $A1 = 2$ |
| Eb2 | $(trig > 0) && (B2 < 1)$ | $5$ | false | false | $A2 = 2$ |
| Eb3 | $(trig > 0) && (B3 < 1)$ | $5$ | false | false | $A3 = 2$ |
| Eb4 | $(trig > 0) && (B4 < 1)$ | $5$ | false | false | $A4 = 2$ |
| Eb5 | $(trig > 0) && (B5 < 1)$ | $5$ | false | false | $A5 = 2$ |
| Eb6 | $(trig > 0) && (B6 < 1)$ | $5$ | false | false | $A6 = 2$ |
| Eb7 | $(trig > 0) && (B7 < 1)$ | $5$ | false | false | $A7 = 2$ |
| Eb8 | $(trig > 0) && (B8 < 1)$ | $5$ | false | false | $A8 = 2$ |
| Eb9 | $(trig > 0) && (B9 < 1)$ | $5$ | false | false | $A9 = 2$ |
| Eb10 | $(trig > 0) && (B10 < 1)$ | $5$ | false | false | $A10 = 2$ |
| Eb11 | $(trig > 0) && (B11 < 1)$ | $5$ | false | false | $A11 = 2$ |
| Eb12 | $(trig > 0) && (B12 < 1)$ | $5$ | false | false | $A12 = 2$ |
| Eb13 | $(trig > 0) && (B13 < 1)$ | $5$ | false | false | $A13 = 2$ |
| Eb14 | $(trig > 0) && (B14 < 1)$ | $5$ | false | false | $A14 = 2$ |
| Eb15 | $(trig > 0) && (B15 < 1)$ | $5$ | false | false | $A15 = 2$ |
| Eb16 | $(trig > 0) && (B16 < 1)$ | $5$ | false | false | $A16 = 2$ |
| Eb17 | $(trig > 0) && (B17 < 1)$ | $5$ | false | false | $A17 = 2$ |
| Eb18 | $(trig > 0) && (B18 < 1)$ | $5$ | false | false | $A18 = 2$ |
| Eb19 | $(trig > 0) && (B19 < 1)$ | $5$ | false | false | $A19 = 2$ |
| Eb20 | $(trig > 0) && (B20 < 1)$ | $5$ | false | false | $A20 = 2$ |
| Eb21 | $(trig > 0) && (B21 < 1)$ | $5$ | false | false | $A21 = 2$ |
| Eb22 | $(trig > 0) && (B22 < 1)$ | $5$ | false | false | $A22 = 2$ |
| Eb23 | $(trig > 0) && (B23 < 1)$ | $5$ | false | false | $A23 = 2$ |
| Eb24 | $(trig > 0) && (B24 < 1)$ | $5$ | false | false | $A24 = 2$ |
| Eb25 | $(trig > 0) && (B25 < 1)$ | $5$ | false | false | $A25 = 2$ |
| Eb26 | $(trig > 0) && (B26 < 1)$ | $5$ | false | false | $A26 = 2$ |
| Eb27 | $(trig > 0) && (B27 < 1)$ | $5$ | false | false | $A27 = 2$ |
| Eb28 | $(trig > 0) && (B28 < 1)$ | $5$ | false | false | $A28 = 2$ |
| Eb29 | $(trig > 0) && (B29 < 1)$ | $5$ | false | false | $A29 = 2$ |
| Eb30 | $(trig > 0) && (B30 < 1)$ | $5$ | false | false | $A30 = 2$ |
| Eb31 | $(trig > 0) && (B31 < 1)$ | $5$ | false | false | $A31 = 2$ |
| Eb32 | $(trig > 0) && (B32 < 1)$ | $5$ | false | false | $A32 = 2$ |
| Eb33 | $(trig > 0) && (B33 < 1)$ | $5$ | false | false | $A33 = 2$ |
| Eb34 | $(trig > 0) && (B34 < 1)$ | $5$ | false | false | $A34 = 2$ |
| Eb35 | $(trig > 0) && (B35 < 1)$ | $5$ | false | false | $A35 = 2$ |
| Eb36 | $(trig > 0) && (B36 < 1)$ | $5$ | false | false | $A36 = 2$ |
| Eb37 | $(trig > 0) && (B37 < 1)$ | $5$ | false | false | $A37 = 2$ |
| Eb38 | $(trig > 0) && (B38 < 1)$ | $5$ | false | false | $A38 = 2$ |
| Eb39 | $(trig > 0) && (B39 < 1)$ | $5$ | false | false | $A39 = 2$ |
| Eb40 | $(trig > 0) && (B40 < 1)$ | $5$ | false | false | $A40 = 2$ |
| Eb41 | $(trig > 0) && (B41 < 1)$ | $5$ | false | false | $A41 = 2$ |
| Eb42 | $(trig > 0) && (B42 < 1)$ | $5$ | false | false | $A42 = 2$ |
| Eb43 | $(trig > 0) && (B43 < 1)$ | $5$ | false | false | $A43 = 2$ |
| Eb44 | $(trig > 0) && (B44 < 1)$ | $5$ | false | false | $A44 = 2$ |
| Eb45 | $(trig > 0) && (B45 < 1)$ | $5$ | false | false | $A45 = 2$ |
| Eb46 | $(trig > 0) && (B46 < 1)$ | $5$ | false | false | $A46 = 2$ |
| Eb47 | $(trig > 0) && (B47 < 1)$ | $5$ | false | false | $A47 = 2$ |
| Eb48 | $(trig > 0) && (B48 < 1)$ | $5$ | false | false | $A48 = 2$ |
| Eb49 | $(trig > 0) && (B49 < 1)$ | $5$ | false | false | $A49 = 2$ |
| Eb50 | $(trig > 0) && (B50 < 1)$ | $5$ | false | false | $A50 = 2$ |
| Eb51 | $(trig > 0) && (B51 < 1)$ | $5$ | false | false | $A51 = 2$ |
| Eb52 | $(trig > 0) && (B52 < 1)$ | $5$ | false | false | $A52 = 2$ |
| Eb53 | $(trig > 0) && (B53 < 1)$ | $5$ | false | false | $A53 = 2$ |
| Eb54 | $(trig > 0) && (B54 < 1)$ | $5$ | false | false | $A54 = 2$ |
| Eb55 | $(trig > 0) && (B55 < 1)$ | $5$ | false | false | $A55 = 2$ |
| Eb56 | $(trig > 0) && (B56 < 1)$ | $5$ | false | false | $A56 = 2$ |
| Eb57 | $(trig > 0) && (B57 < 1)$ | $5$ | false | false | $A57 = 2$ |
| Eb58 | $(trig > 0) && (B58 < 1)$ | $5$ | false | false | $A58 = 2$ |
| Eb59 | $(trig > 0) && (B59 < 1)$ | $5$ | false | false | $A59 = 2$ |
| Eb60 | $(trig > 0) && (B60 < 1)$ | $5$ | false | false | $A60 = 2$ |
| Eb61 | $(trig > 0) && (B61 < 1)$ | $5$ | false | false | $A61 = 2$ |
| Eb62 | $(trig > 0) && (B62 < 1)$ | $5$ | false | false | $A62 = 2$ |
| Eb63 | $(trig > 0) && (B63 < 1)$ | $5$ | false | false | $A63 = 2$ |
| Eb64 | $(trig > 0) && (B64 < 1)$ | $5$ | false | false | $A64 = 2$ |
| Eb65 | $(trig > 0) && (B65 < 1)$ | $5$ | false | false | $A65 = 2$ |
| Eb66 | $(trig > 0) && (B66 < 1)$ | $5$ | false | false | $A66 = 2$ |
| Eb67 | $(trig > 0) && (B67 < 1)$ | $5$ | false | false | $A67 = 2$ |
| Eb68 | $(trig > 0) && (B68 < 1)$ | $5$ | false | false | $A68 = 2$ |
| Eb69 | $(trig > 0) && (B69 < 1)$ | $5$ | false | false | $A69 = 2$ |
| Eb70 | $(trig > 0) && (B70 < 1)$ | $5$ | false | false | $A70 = 2$ |
| Eb71 | $(trig > 0) && (B71 < 1)$ | $5$ | false | false | $A71 = 2$ |
| Eb72 | $(trig > 0) && (B72 < 1)$ | $5$ | false | false | $A72 = 2$ |
| Eb73 | $(trig > 0) && (B73 < 1)$ | $5$ | false | false | $A73 = 2$ |
| Eb74 | $(trig > 0) && (B74 < 1)$ | $5$ | false | false | $A74 = 2$ |
| Eb75 | $(trig > 0) && (B75 < 1)$ | $5$ | false | false | $A75 = 2$ |
| Eb76 | $(trig > 0) && (B76 < 1)$ | $5$ | false | false | $A76 = 2$ |
| Eb77 | $(trig > 0) && (B77 < 1)$ | $5$ | false | false | $A77 = 2$ |
| Eb78 | $(trig > 0) && (B78 < 1)$ | $5$ | false | false | $A78 = 2$ |
| Eb79 | $(trig > 0) && (B79 < 1)$ | $5$ | false | false | $A79 = 2$ |
| Eb80 | $(trig > 0) && (B80 < 1)$ | $5$ | false | false | $A80 = 2$ |
| Eb81 | $(trig > 0) && (B81 < 1)$ | $5$ | false | false | $A81 = 2$ |
| Eb82 | $(trig > 0) && (B82 < 1)$ | $5$ | false | false | $A82 = 2$ |
| Eb83 | $(trig > 0) && (B83 < 1)$ | $5$ | false | false | $A83 = 2$ |
| Eb84 | $(trig > 0) && (B84 < 1)$ | $5$ | false | false | $A84 = 2$ |
| Eb85 | $(trig > 0) && (B85 < 1)$ | $5$ | false | false | $A85 = 2$ |
| Eb86 | $(trig > 0) && (B86 < 1)$ | $5$ | false | false | $A86 = 2$ |
| Eb87 | $(trig > 0) && (B87 < 1)$ | $5$ | false | false | $A87 = 2$ |
| Eb88 | $(trig > 0) && (B88 < 1)$ | $5$ | false | false | $A88 = 2$ |
| Eb89 | $(trig > 0) && (B89 < 1)$ | $5$ | false | false | $A89 = 2$ |
| Eb90 | $(trig > 0) && (B90 < 1)$ | $5$ | false | false | $A90 = 2$ |
| Eb91 | $(trig > 0) && (B91 < 1)$ | $5$ | false | false | $A91 = 2$ |
| Eb92 | $(trig > 0) && (B92 < 1)$ | $5$ | false | false | $A92 = 2$ |
| Eb93 | $(trig > 0) && (B93 < 1)$ | $5$ | false | false | $A93 = 2$ |
| Eb94 | $(trig > 0) && (B94 < 1)$ | $5$ | false | false | $A94 = 2$ |
| Eb95 | $(trig > 0) && (B95 < 1)$ | $5$ | false | false | $A95 = 2$ |
| Eb96 | $(trig > 0) && (B96 < 1)$ | $5$ | false | false | $A96 = 2$ |
| Eb97 | $(trig > 0) && (B97 < 1)$ | $5$ | false | false | $A97 = 2$ |
| Eb98 | $(trig > 0) && (B98 < 1)$ | $5$ | false | false | $A98 = 2$ |
| Eb99 | $(trig > 0) && (B99 < 1)$ | $5$ | false | false | $A99 = 2$ |
| _E0 | $abs(Asum - Bsum) > 100$ | (unset) | true | true | $toodiverged = 1$ |]


There are 3 rules:

[{width:30em,margin: 1em auto}|  *Type*  |  *Variable*  |  *Formula*  |
| Assignment | Asum | $A0 + A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8 + A9 + A10 + A11 + A12 + A13 + A14 + A15 + A16 + A17 + A18 + A19 + A20 + A21 + A22 + A23 + A24 + A25 + A26 + A27 + A28 + A29 + A30 + A31 + A32 + A33 + A34 + A35 + A36 + A37 + A38 + A39 + A40 + A41 + A42 + A43 + A44 + A45 + A46 + A47 + A48 + A49 + A50 + A51 + A52 + A53 + A54 + A55 + A56 + A57 + A58 + A59 + A60 + A61 + A62 + A63 + A64 + A65 + A66 + A67 + A68 + A69 + A70 + A71 + A72 + A73 + A74 + A75 + A76 + A77 + A78 + A79 + A80 + A81 + A82 + A83 + A84 + A85 + A86 + A87 + A88 + A89 + A90 + A91 + A92 + A93 + A94 + A95 + A96 + A97 + A98 + A99$ |
| Assignment | Bsum | $B0 + B1 + B2 + B3 + B4 + B5 + B6 + B7 + B8 + B9 + B10 + B11 + B12 + B13 + B14 + B15 + B16 + B17 + B18 + B19 + B20 + B21 + B22 + B23 + B24 + B25 + B26 + B27 + B28 + B29 + B30 + B31 + B32 + B33 + B34 + B35 + B36 + B37 + B38 + B39 + B40 + B41 + B42 + B43 + B44 + B45 + B46 + B47 + B48 + B49 + B50 + B51 + B52 + B53 + B54 + B55 + B56 + B57 + B58 + B59 + B60 + B61 + B62 + B63 + B64 + B65 + B66 + B67 + B68 + B69 + B70 + B71 + B72 + B73 + B74 + B75 + B76 + B77 + B78 + B79 + B80 + B81 + B82 + B83 + B84 + B85 + B86 + B87 + B88 + B89 + B90 + B91 + B92 + B93 + B94 + B95 + B96 + B97 + B98 + B99$ |
| Assignment | Allsum | $Asum + Bsum$ |]

The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter trig | $1$ | constant |
| Initial value of parameter A0 | $0$ | variable |
| Initial value of parameter A1 | $0$ | variable |
| Initial value of parameter A2 | $0$ | variable |
| Initial value of parameter A3 | $0$ | variable |
| Initial value of parameter A4 | $0$ | variable |
| Initial value of parameter A5 | $0$ | variable |
| Initial value of parameter A6 | $0$ | variable |
| Initial value of parameter A7 | $0$ | variable |
| Initial value of parameter A8 | $0$ | variable |
| Initial value of parameter A9 | $0$ | variable |
| Initial value of parameter A10 | $0$ | variable |
| Initial value of parameter A11 | $0$ | variable |
| Initial value of parameter A12 | $0$ | variable |
| Initial value of parameter A13 | $0$ | variable |
| Initial value of parameter A14 | $0$ | variable |
| Initial value of parameter A15 | $0$ | variable |
| Initial value of parameter A16 | $0$ | variable |
| Initial value of parameter A17 | $0$ | variable |
| Initial value of parameter A18 | $0$ | variable |
| Initial value of parameter A19 | $0$ | variable |
| Initial value of parameter A20 | $0$ | variable |
| Initial value of parameter A21 | $0$ | variable |
| Initial value of parameter A22 | $0$ | variable |
| Initial value of parameter A23 | $0$ | variable |
| Initial value of parameter A24 | $0$ | variable |
| Initial value of parameter A25 | $0$ | variable |
| Initial value of parameter A26 | $0$ | variable |
| Initial value of parameter A27 | $0$ | variable |
| Initial value of parameter A28 | $0$ | variable |
| Initial value of parameter A29 | $0$ | variable |
| Initial value of parameter A30 | $0$ | variable |
| Initial value of parameter A31 | $0$ | variable |
| Initial value of parameter A32 | $0$ | variable |
| Initial value of parameter A33 | $0$ | variable |
| Initial value of parameter A34 | $0$ | variable |
| Initial value of parameter A35 | $0$ | variable |
| Initial value of parameter A36 | $0$ | variable |
| Initial value of parameter A37 | $0$ | variable |
| Initial value of parameter A38 | $0$ | variable |
| Initial value of parameter A39 | $0$ | variable |
| Initial value of parameter A40 | $0$ | variable |
| Initial value of parameter A41 | $0$ | variable |
| Initial value of parameter A42 | $0$ | variable |
| Initial value of parameter A43 | $0$ | variable |
| Initial value of parameter A44 | $0$ | variable |
| Initial value of parameter A45 | $0$ | variable |
| Initial value of parameter A46 | $0$ | variable |
| Initial value of parameter A47 | $0$ | variable |
| Initial value of parameter A48 | $0$ | variable |
| Initial value of parameter A49 | $0$ | variable |
| Initial value of parameter A50 | $0$ | variable |
| Initial value of parameter A51 | $0$ | variable |
| Initial value of parameter A52 | $0$ | variable |
| Initial value of parameter A53 | $0$ | variable |
| Initial value of parameter A54 | $0$ | variable |
| Initial value of parameter A55 | $0$ | variable |
| Initial value of parameter A56 | $0$ | variable |
| Initial value of parameter A57 | $0$ | variable |
| Initial value of parameter A58 | $0$ | variable |
| Initial value of parameter A59 | $0$ | variable |
| Initial value of parameter A60 | $0$ | variable |
| Initial value of parameter A61 | $0$ | variable |
| Initial value of parameter A62 | $0$ | variable |
| Initial value of parameter A63 | $0$ | variable |
| Initial value of parameter A64 | $0$ | variable |
| Initial value of parameter A65 | $0$ | variable |
| Initial value of parameter A66 | $0$ | variable |
| Initial value of parameter A67 | $0$ | variable |
| Initial value of parameter A68 | $0$ | variable |
| Initial value of parameter A69 | $0$ | variable |
| Initial value of parameter A70 | $0$ | variable |
| Initial value of parameter A71 | $0$ | variable |
| Initial value of parameter A72 | $0$ | variable |
| Initial value of parameter A73 | $0$ | variable |
| Initial value of parameter A74 | $0$ | variable |
| Initial value of parameter A75 | $0$ | variable |
| Initial value of parameter A76 | $0$ | variable |
| Initial value of parameter A77 | $0$ | variable |
| Initial value of parameter A78 | $0$ | variable |
| Initial value of parameter A79 | $0$ | variable |
| Initial value of parameter A80 | $0$ | variable |
| Initial value of parameter A81 | $0$ | variable |
| Initial value of parameter A82 | $0$ | variable |
| Initial value of parameter A83 | $0$ | variable |
| Initial value of parameter A84 | $0$ | variable |
| Initial value of parameter A85 | $0$ | variable |
| Initial value of parameter A86 | $0$ | variable |
| Initial value of parameter A87 | $0$ | variable |
| Initial value of parameter A88 | $0$ | variable |
| Initial value of parameter A89 | $0$ | variable |
| Initial value of parameter A90 | $0$ | variable |
| Initial value of parameter A91 | $0$ | variable |
| Initial value of parameter A92 | $0$ | variable |
| Initial value of parameter A93 | $0$ | variable |
| Initial value of parameter A94 | $0$ | variable |
| Initial value of parameter A95 | $0$ | variable |
| Initial value of parameter A96 | $0$ | variable |
| Initial value of parameter A97 | $0$ | variable |
| Initial value of parameter A98 | $0$ | variable |
| Initial value of parameter A99 | $0$ | variable |
| Initial value of parameter B0 | $0$ | variable |
| Initial value of parameter B1 | $0$ | variable |
| Initial value of parameter B2 | $0$ | variable |
| Initial value of parameter B3 | $0$ | variable |
| Initial value of parameter B4 | $0$ | variable |
| Initial value of parameter B5 | $0$ | variable |
| Initial value of parameter B6 | $0$ | variable |
| Initial value of parameter B7 | $0$ | variable |
| Initial value of parameter B8 | $0$ | variable |
| Initial value of parameter B9 | $0$ | variable |
| Initial value of parameter B10 | $0$ | variable |
| Initial value of parameter B11 | $0$ | variable |
| Initial value of parameter B12 | $0$ | variable |
| Initial value of parameter B13 | $0$ | variable |
| Initial value of parameter B14 | $0$ | variable |
| Initial value of parameter B15 | $0$ | variable |
| Initial value of parameter B16 | $0$ | variable |
| Initial value of parameter B17 | $0$ | variable |
| Initial value of parameter B18 | $0$ | variable |
| Initial value of parameter B19 | $0$ | variable |
| Initial value of parameter B20 | $0$ | variable |
| Initial value of parameter B21 | $0$ | variable |
| Initial value of parameter B22 | $0$ | variable |
| Initial value of parameter B23 | $0$ | variable |
| Initial value of parameter B24 | $0$ | variable |
| Initial value of parameter B25 | $0$ | variable |
| Initial value of parameter B26 | $0$ | variable |
| Initial value of parameter B27 | $0$ | variable |
| Initial value of parameter B28 | $0$ | variable |
| Initial value of parameter B29 | $0$ | variable |
| Initial value of parameter B30 | $0$ | variable |
| Initial value of parameter B31 | $0$ | variable |
| Initial value of parameter B32 | $0$ | variable |
| Initial value of parameter B33 | $0$ | variable |
| Initial value of parameter B34 | $0$ | variable |
| Initial value of parameter B35 | $0$ | variable |
| Initial value of parameter B36 | $0$ | variable |
| Initial value of parameter B37 | $0$ | variable |
| Initial value of parameter B38 | $0$ | variable |
| Initial value of parameter B39 | $0$ | variable |
| Initial value of parameter B40 | $0$ | variable |
| Initial value of parameter B41 | $0$ | variable |
| Initial value of parameter B42 | $0$ | variable |
| Initial value of parameter B43 | $0$ | variable |
| Initial value of parameter B44 | $0$ | variable |
| Initial value of parameter B45 | $0$ | variable |
| Initial value of parameter B46 | $0$ | variable |
| Initial value of parameter B47 | $0$ | variable |
| Initial value of parameter B48 | $0$ | variable |
| Initial value of parameter B49 | $0$ | variable |
| Initial value of parameter B50 | $0$ | variable |
| Initial value of parameter B51 | $0$ | variable |
| Initial value of parameter B52 | $0$ | variable |
| Initial value of parameter B53 | $0$ | variable |
| Initial value of parameter B54 | $0$ | variable |
| Initial value of parameter B55 | $0$ | variable |
| Initial value of parameter B56 | $0$ | variable |
| Initial value of parameter B57 | $0$ | variable |
| Initial value of parameter B58 | $0$ | variable |
| Initial value of parameter B59 | $0$ | variable |
| Initial value of parameter B60 | $0$ | variable |
| Initial value of parameter B61 | $0$ | variable |
| Initial value of parameter B62 | $0$ | variable |
| Initial value of parameter B63 | $0$ | variable |
| Initial value of parameter B64 | $0$ | variable |
| Initial value of parameter B65 | $0$ | variable |
| Initial value of parameter B66 | $0$ | variable |
| Initial value of parameter B67 | $0$ | variable |
| Initial value of parameter B68 | $0$ | variable |
| Initial value of parameter B69 | $0$ | variable |
| Initial value of parameter B70 | $0$ | variable |
| Initial value of parameter B71 | $0$ | variable |
| Initial value of parameter B72 | $0$ | variable |
| Initial value of parameter B73 | $0$ | variable |
| Initial value of parameter B74 | $0$ | variable |
| Initial value of parameter B75 | $0$ | variable |
| Initial value of parameter B76 | $0$ | variable |
| Initial value of parameter B77 | $0$ | variable |
| Initial value of parameter B78 | $0$ | variable |
| Initial value of parameter B79 | $0$ | variable |
| Initial value of parameter B80 | $0$ | variable |
| Initial value of parameter B81 | $0$ | variable |
| Initial value of parameter B82 | $0$ | variable |
| Initial value of parameter B83 | $0$ | variable |
| Initial value of parameter B84 | $0$ | variable |
| Initial value of parameter B85 | $0$ | variable |
| Initial value of parameter B86 | $0$ | variable |
| Initial value of parameter B87 | $0$ | variable |
| Initial value of parameter B88 | $0$ | variable |
| Initial value of parameter B89 | $0$ | variable |
| Initial value of parameter B90 | $0$ | variable |
| Initial value of parameter B91 | $0$ | variable |
| Initial value of parameter B92 | $0$ | variable |
| Initial value of parameter B93 | $0$ | variable |
| Initial value of parameter B94 | $0$ | variable |
| Initial value of parameter B95 | $0$ | variable |
| Initial value of parameter B96 | $0$ | variable |
| Initial value of parameter B97 | $0$ | variable |
| Initial value of parameter B98 | $0$ | variable |
| Initial value of parameter B99 | $0$ | variable |
| Initial value of parameter Asum | $A0 + A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8 + A9 + A10 + A11 + A12 + A13 + A14 + A15 + A16 + A17 + A18 + A19 + A20 + A21 + A22 + A23 + A24 + A25 + A26 + A27 + A28 + A29 + A30 + A31 + A32 + A33 + A34 + A35 + A36 + A37 + A38 + A39 + A40 + A41 + A42 + A43 + A44 + A45 + A46 + A47 + A48 + A49 + A50 + A51 + A52 + A53 + A54 + A55 + A56 + A57 + A58 + A59 + A60 + A61 + A62 + A63 + A64 + A65 + A66 + A67 + A68 + A69 + A70 + A71 + A72 + A73 + A74 + A75 + A76 + A77 + A78 + A79 + A80 + A81 + A82 + A83 + A84 + A85 + A86 + A87 + A88 + A89 + A90 + A91 + A92 + A93 + A94 + A95 + A96 + A97 + A98 + A99$ | variable |
| Initial value of parameter Bsum | $B0 + B1 + B2 + B3 + B4 + B5 + B6 + B7 + B8 + B9 + B10 + B11 + B12 + B13 + B14 + B15 + B16 + B17 + B18 + B19 + B20 + B21 + B22 + B23 + B24 + B25 + B26 + B27 + B28 + B29 + B30 + B31 + B32 + B33 + B34 + B35 + B36 + B37 + B38 + B39 + B40 + B41 + B42 + B43 + B44 + B45 + B46 + B47 + B48 + B49 + B50 + B51 + B52 + B53 + B54 + B55 + B56 + B57 + B58 + B59 + B60 + B61 + B62 + B63 + B64 + B65 + B66 + B67 + B68 + B69 + B70 + B71 + B72 + B73 + B74 + B75 + B76 + B77 + B78 + B79 + B80 + B81 + B82 + B83 + B84 + B85 + B86 + B87 + B88 + B89 + B90 + B91 + B92 + B93 + B94 + B95 + B96 + B97 + B98 + B99$ | variable |
| Initial value of parameter Allsum | $Asum + Bsum$ | variable |
| Initial value of parameter toodiverged | $0$ | variable |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
