/**
 * @file    TestListOfFunctionTerms.cpp
 * @brief   TestListOfFunctionTerms unit tests
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
#include <sbml/packages/qual/common/QualExtensionTypes.h>
#include <sbml/extension/SBMLExtensionRegistry.h>
#include <sbml/SBMLTypeCodes.h>
#include <string>

/** @cond doxygenIgnored */

using namespace std;
LIBSBML_CPP_NAMESPACE_USE

/** @endcond doxygenIgnored */


CK_CPPSTART

static ListOfFunctionTerms* G; 
static QualPkgNamespaces* GNS;

void
ListOfFunctionTermsTest_setup (void)
{
  GNS = new QualPkgNamespaces();
  G = new ListOfFunctionTerms(GNS);
  
  if (G == NULL)
  {
    fail("Failed to create a ListOfFunctionTerms object");
  }
}


void
ListOfFunctionTermsTest_teardown (void)
{
  delete G;
  delete GNS;
}


START_TEST (test_ListOfFunctionTerms_create)
{
  fail_unless(G->isSetDefaultTerm() == false);
  fail_unless(G->size() == 0);
}
END_TEST


START_TEST (test_ListOfFunctionTerms_functionTerms)
{
  fail_unless(G->isSetDefaultTerm() == false);
  fail_unless(G->size() == 0);

  FunctionTerm *ft = new FunctionTerm(GNS);
  ASTNode * ast = SBML_parseL3Formula("geq(1,2)");
  ft->setResultLevel(1);
  ft->setMath(ast);
  
  fail_unless(G->appendAndOwn(ft) == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetDefaultTerm() == false);
  fail_unless(G->size() == 1);
}
END_TEST


START_TEST (test_ListOfFunctionTerms_defaultTerm)
{
  fail_unless(G->isSetDefaultTerm() == false);
  fail_unless(G->size() == 0);

  DefaultTerm *dt = new DefaultTerm(GNS);
  dt->setResultLevel(1);

  fail_unless(G->setDefaultTerm(dt) == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetDefaultTerm() == true);
  fail_unless(G->size() == 0);
}
END_TEST


START_TEST(test_ListOfFunctionTerms_copy)
{
  FunctionTerm *ft = new FunctionTerm(GNS);
  ASTNode * ast = SBML_parseL3Formula("geq(1,2)");
  ft->setResultLevel(1);
  ft->setMath(ast);
  DefaultTerm *dt = new DefaultTerm(GNS);
  dt->setResultLevel(1);
  G->appendAndOwn(ft);
  G->setDefaultTerm(dt);

  ListOfFunctionTerms *g2 = new ListOfFunctionTerms(*G);

  fail_unless(g2->isSetDefaultTerm() == true);
  fail_unless(g2->size()   == 1);
 
  delete g2;
}
END_TEST


START_TEST(test_ListOfFunctionTerms_assignment)
{
  FunctionTerm *ft = new FunctionTerm(GNS);
  ASTNode * ast = SBML_parseL3Formula("geq(1,2)");
  ft->setResultLevel(1);
  ft->setMath(ast);
  DefaultTerm *dt = new DefaultTerm(GNS);
  dt->setResultLevel(1);
  G->appendAndOwn(ft);
  G->setDefaultTerm(dt);

  ListOfFunctionTerms* g2 = new ListOfFunctionTerms();

  (*g2) = (*G);

  fail_unless(g2->isSetDefaultTerm() == true);
  fail_unless(g2->size()   == 1);

  delete g2;
}
END_TEST


START_TEST(test_ListOfFunctionTerms_clone)
{
  FunctionTerm *ft = new FunctionTerm(GNS);
  ASTNode * ast = SBML_parseL3Formula("geq(1,2)");
  ft->setResultLevel(1);
  ft->setMath(ast);
  DefaultTerm *dt = new DefaultTerm(GNS);
  dt->setResultLevel(1);
  G->appendAndOwn(ft);
  G->setDefaultTerm(dt);

  ListOfFunctionTerms* g2 = G->clone();
  
  fail_unless(g2->isSetDefaultTerm() == true);
  fail_unless(g2->size()   == 1);
  
  delete g2;
}
END_TEST


Suite *
create_suite_ListOfFunctionTerms (void)
{
  Suite *suite = suite_create("ListOfFunctionTerms");
  TCase *tcase = tcase_create("ListOfFunctionTerms");

  tcase_add_checked_fixture(tcase, ListOfFunctionTermsTest_setup, ListOfFunctionTermsTest_teardown);
 
  tcase_add_test( tcase, test_ListOfFunctionTerms_create         );

  tcase_add_test( tcase, test_ListOfFunctionTerms_functionTerms );
  tcase_add_test( tcase, test_ListOfFunctionTerms_defaultTerm        );

  tcase_add_test( tcase, test_ListOfFunctionTerms_copy            );
  tcase_add_test( tcase, test_ListOfFunctionTerms_assignment      );
  tcase_add_test( tcase, test_ListOfFunctionTerms_clone           );

  suite_add_tcase(suite, tcase);

  return suite;
}


CK_CPPEND
