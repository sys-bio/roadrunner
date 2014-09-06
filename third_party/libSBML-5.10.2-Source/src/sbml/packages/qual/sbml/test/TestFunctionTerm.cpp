/**
 * @file    TestFunctionTerm.cpp
 * @brief   TestFunctionTerm unit tests
 * @author  Sarah Keating
 *
 * $Id: $
 * $HeadURL: $
 *
 * <!--------------------------------------------------------------------------
 * This file is part of libSBML.  Please visit http://sbml.org for more
 * information about SBML, and the latest version of libSBML.
 *
 * Copyright (C) 2013-2014 jointly by the following organizations:
 *     1. California Institute of Technology, Pasadena, CA, USA
 *     2. EMBL European Bioinformatics Institute (EMBL-EBI), Hinxton, UK
 *     3. University of Heidelberg, Heidelberg, Germany
 * 
 * Copyright (C) 2009-2011 jointly by the following organizations: 
 *     1. California Institute of Technology, Pasadena, CA, USA
 *     2. EMBL European Bioinformatics Institute (EMBL-EBI), Hinxton, UK
 *  
 * Copyright (C) 2006-2008 by the California Institute of Technology,
 *     Pasadena, CA, USA 
 *  
 * Copyright (C) 2002-2005 jointly by the following organizations: 
 *     1. California Institute of Technology, Pasadena, CA, USA
 *     2. Japan Science and Technology Agency, Japan
 * 
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation.  A copy of the license agreement is provided
 * in the file ResultLeveld "LICENSE.txt" included with this software distribution
 * and also available online as http://sbml.org/software/libsbml/license.html
 * ---------------------------------------------------------------------- -->*/

#include <limits>

#include <iostream>
#include <check.h>
#include <sbml/common/extern.h>
#include <sbml/math/L3Parser.h>
#include <sbml/math/FormulaFormatter.h>
#include <sbml/packages/qual/common/QualExtensionTypes.h>
#include <sbml/extension/SBMLExtensionRegistry.h>
#include <sbml/SBMLTypeCodes.h>
#include <string>

/** @cond doxygenIgnored */

using namespace std;
LIBSBML_CPP_NAMESPACE_USE

/** @endcond doxygenIgnored */


CK_CPPSTART

static FunctionTerm* G; 
static QualPkgNamespaces* GNS;

void
FunctionTermTest_setup (void)
{
  GNS = new QualPkgNamespaces();
  G = new FunctionTerm(GNS);
  
  if (G == NULL)
  {
    fail("Failed to create a FunctionTerm object");
  }
}


void
FunctionTermTest_teardown (void)
{
  delete G;
  delete GNS;
}


START_TEST (test_FunctionTerm_create)
{
  fail_unless(G->isSetResultLevel() == false);
  fail_unless(G->isSetMath()        == false);

}
END_TEST


START_TEST (test_FunctionTerm_resultLevel)
{
  fail_unless(G->isSetResultLevel() == false);

  fail_unless(G->setResultLevel(1) == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetResultLevel() == true);
  fail_unless(G->getResultLevel() == 1);

  fail_unless(G->unsetResultLevel() == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetResultLevel() == false);
}
END_TEST


START_TEST (test_FunctionTerm_math)
{
  fail_unless(G->isSetMath() == false);

  ASTNode * ast = SBML_parseL3Formula("geq(1,2)");

  fail_unless(G->setMath(ast) == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetMath() == true);
  fail_unless(!strcmp(SBML_formulaToString(G->getMath()), "geq(1, 2)"));

  fail_unless(G->unsetMath() == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetMath() == false);
}
END_TEST


START_TEST(test_FunctionTerm_copy)
{
  G->setResultLevel(2);

  FunctionTerm *g2 = new FunctionTerm(*G);

  fail_unless(g2->isSetResultLevel() == true);
  fail_unless(g2->isSetMath()        == false);
  fail_unless(g2->getResultLevel()   == 2);
 
  delete g2;
}
END_TEST


START_TEST(test_FunctionTerm_assignment)
{
  G->setResultLevel(2);
  FunctionTerm* g2 = new FunctionTerm();

  (*g2) = (*G);

  fail_unless(g2->isSetResultLevel() == true);
  fail_unless(g2->isSetMath()        == false);
  fail_unless(g2->getResultLevel()   == 2);

  delete g2;
}
END_TEST


START_TEST(test_FunctionTerm_clone)
{
  G->setResultLevel(2);
  FunctionTerm* g2 = G->clone();
  
  fail_unless(g2->isSetResultLevel() == true);
  fail_unless(g2->isSetMath()        == false);
  fail_unless(g2->getResultLevel()   == 2);
  
  delete g2;
}
END_TEST


Suite *
create_suite_FunctionTerm (void)
{
  Suite *suite = suite_create("FunctionTerm");
  TCase *tcase = tcase_create("FunctionTerm");

  tcase_add_checked_fixture(tcase, FunctionTermTest_setup, FunctionTermTest_teardown);
 
  tcase_add_test( tcase, test_FunctionTerm_create         );

  tcase_add_test( tcase, test_FunctionTerm_resultLevel );
  tcase_add_test( tcase, test_FunctionTerm_math        );

  tcase_add_test( tcase, test_FunctionTerm_copy            );
  tcase_add_test( tcase, test_FunctionTerm_assignment      );
  tcase_add_test( tcase, test_FunctionTerm_clone           );

  suite_add_tcase(suite, tcase);

  return suite;
}


CK_CPPEND
