(*

category:        Test
synopsis:        Initial assignments that test that the appropriate value for avogadro is being used.
componentTags:   CSymbolAvogadro, InitialAssignment, Parameter
testTags:        InitialValueReassigned
testType:        TimeCourse
levels:          3.1, 3.2
generatedBy:     Analytic
packagesPresent: 

Since 1865, people have been calculating avogadro's number with increasing accuracy.  In 2010, when the SBML Level 3 specification came out, the most accurate value anyone had calculated was 6.02214179e23, and this value was taken as canonical for all SBML files.  Since (and before) then, avogadro has taken other values.  However, none of them should be used.  This file checks that you use the value of avogadro from the correct year, by dividing the literal value of avogadro in different years by the csymbol 'avogadro'.  If you fail this test, but get a value of '1' for one of your parameters, you are using the value of avogadro established in that year, and not as it had been determined to be in 2010.

Another possibility is that the test itself is going wrong because your program has truncated the values from which avogadro is being subtracted:  if you get the same result for 'a_truncated' as you get for 'a_2006', this probably means you're either using the infix output of libsbml version 5.14.0 or earlier, which truncated all 'e-notation' values to seven significant digits, or a similar scheme that does the same thing.

I don't really think that you're likely to be accidentally using the value of avogadro from 1865, but those values are provided here to sate your historical curiosity.

Values before and including 2001 were found in http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.408.199&rep=rep1&type=pdf, while values after that were found through direct citations, and by using the Internet Archive's copies of CODATA's recommended value for Avogadro:  http://web.archive.org/web/20040215162824*/http://physics.nist.gov/cgi-bin/cuu/Value?na

The model contains:
* 41 parameters (a_sbml, a_truncated, a_1865, a_1873, a_1890a, a_1890b, a_1901, a_1903, a_1904, a_1908, a_1909, a_1914a, a_1914b, a_1915, a_1917, a_1923, a_1924, a_1929, a_1931, a_1941, a_1945, a_1948, a_1949, a_1951, a_1965a, a_1965b, a_1973, a_1974, a_1987a, a_1987b, a_1992, a_1994, a_1995, a_1999, a_2000, a_2001, a_2002, a_2003, a_2006, a_2011, a_2015)
The initial conditions are as follows:

[{width:35em,margin: 1em auto}|       | *Value* | *Constant* |
| Initial value of parameter a_sbml | $6.02214179e23 / avogadro$ | constant |
| Initial value of parameter a_truncated | $6.022142e23 / avogadro$ | constant |
| Initial value of parameter a_1865 | $7.2e24 / avogadro$ | constant |
| Initial value of parameter a_1873 | $1.1e24 / avogadro$ | constant |
| Initial value of parameter a_1890a | $7e23 / avogadro$ | constant |
| Initial value of parameter a_1890b | $6.08e23 / avogadro$ | constant |
| Initial value of parameter a_1901 | $6.16e23 / avogadro$ | constant |
| Initial value of parameter a_1903 | $9.3e23 / avogadro$ | constant |
| Initial value of parameter a_1904 | $8.7e23 / avogadro$ | constant |
| Initial value of parameter a_1908 | $6e23 / avogadro$ | constant |
| Initial value of parameter a_1909 | $6.16e23 / avogadro$ | constant |
| Initial value of parameter a_1914a | $6e23 / avogadro$ | constant |
| Initial value of parameter a_1914b | $5.91e23 / avogadro$ | constant |
| Initial value of parameter a_1915 | $6.06e23 / avogadro$ | constant |
| Initial value of parameter a_1917 | $6.064e23 / avogadro$ | constant |
| Initial value of parameter a_1923 | $5.9e23 / avogadro$ | constant |
| Initial value of parameter a_1924 | $6.004e23 / avogadro$ | constant |
| Initial value of parameter a_1929 | $6.0644e23 / avogadro$ | constant |
| Initial value of parameter a_1931 | $6.019e23 / avogadro$ | constant |
| Initial value of parameter a_1941 | $6.02283e23 / avogadro$ | constant |
| Initial value of parameter a_1945 | $6.02338e23 / avogadro$ | constant |
| Initial value of parameter a_1948 | $6.0232e23 / avogadro$ | constant |
| Initial value of parameter a_1949 | $6.02403e23 / avogadro$ | constant |
| Initial value of parameter a_1951 | $6.02544e23 / avogadro$ | constant |
| Initial value of parameter a_1965a | $6.022088e23 / avogadro$ | constant |
| Initial value of parameter a_1965b | $6.02252e23 / avogadro$ | constant |
| Initial value of parameter a_1973 | $6.022045e23 / avogadro$ | constant |
| Initial value of parameter a_1974 | $6.0220943e23 / avogadro$ | constant |
| Initial value of parameter a_1987a | $6.022134e23 / avogadro$ | constant |
| Initial value of parameter a_1987b | $6.0221367e23 / avogadro$ | constant |
| Initial value of parameter a_1992 | $6.0221363e23 / avogadro$ | constant |
| Initial value of parameter a_1994 | $6.0221379e23 / avogadro$ | constant |
| Initial value of parameter a_1995 | $6.0221365e23 / avogadro$ | constant |
| Initial value of parameter a_1999 | $6.022155e23 / avogadro$ | constant |
| Initial value of parameter a_2000 | $6.02214199e23 / avogadro$ | constant |
| Initial value of parameter a_2001 | $6.0221339e23 / avogadro$ | constant |
| Initial value of parameter a_2002 | $6.0221415e23 / avogadro$ | constant |
| Initial value of parameter a_2003 | $6.0221353e23 / avogadro$ | constant |
| Initial value of parameter a_2006 | $6.02214179e23 / avogadro$ | constant |
| Initial value of parameter a_2011 | $6.02214129e23 / avogadro$ | constant |
| Initial value of parameter a_2015 | $6.022140857e23 / avogadro$ | constant |]

Note: The test data for this model was generated from an analytical
solution of the system of equations.

*)
