/**
 * @file    TestQualExtension.cpp
 * @brief   TestQualExtension unit tests
 * @author  Akiya Jouraku
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
 * in the file named "LICENSE.txt" included with this software distribution
 * and also available online as http://sbml.org/software/libsbml/license.html
 * ---------------------------------------------------------------------- -->*/

#include <limits>

#include <iostream>
#include <check.h>
#include <sbml/common/extern.h>
#include <sbml/packages/qual/common/QualExtensionTypes.h>
#include <sbml/extension/SBMLExtensionRegistry.h>
#include <sbml/SBMLTypeCodes.h>
#include <string>

/** @cond doxygenIgnored */

using namespace std;
LIBSBML_CPP_NAMESPACE_USE

/** @endcond doxygenIgnored */


CK_CPPSTART

static QualExtension* G; 
static QualPkgNamespaces* GNS;
static string QUAL_XMLNS_L3V1V1;
static string CORE_XMLNS_L2V4;
static string QUAL_PACKAGE_NAME;

void
QualExtensionTest_setup (void)
{
  try
  {
    G = new QualExtension();
    GNS = new QualPkgNamespaces();
    QUAL_PACKAGE_NAME = G->getName();
    QUAL_XMLNS_L3V1V1 = GNS->getURI();
    CORE_XMLNS_L2V4 = SBMLNamespaces::getSBMLNamespaceURI(2, 4);
  }
  catch(...)
  {
    fail("Failed to create a QualExtension object");
  }
}


void
QualExtensionTest_teardown (void)
{
  delete G;
  delete GNS;
}


START_TEST (test_QualExtension_getName)
{
  fail_unless(G->getName() == "qual");
  fail_unless(G->getName() == QUAL_PACKAGE_NAME);
}
END_TEST


START_TEST (test_QualExtension_getURI)
{
  fail_unless(G->getURI(3,1,1) == QUAL_XMLNS_L3V1V1);
  fail_unless(G->getURI(2,1,1) == "");
  fail_unless(G->getURI(4,1,1) == "");
}
END_TEST


START_TEST (test_QualExtension_getLevelVersion)
{
  fail_unless(G->getLevel(QUAL_XMLNS_L3V1V1) == 3);
  fail_unless(G->getLevel(CORE_XMLNS_L2V4)             == 0);
  fail_unless(G->getLevel("")                          == 0);

  fail_unless(G->getVersion(QUAL_XMLNS_L3V1V1) == 1);
  fail_unless(G->getVersion(CORE_XMLNS_L2V4)             == 0);
  fail_unless(G->getVersion("")                          == 0);

  fail_unless(G->getPackageVersion(QUAL_XMLNS_L3V1V1) == 1);
  fail_unless(G->getPackageVersion(CORE_XMLNS_L2V4)             == 0);
  fail_unless(G->getPackageVersion("")                          == 0);
}
END_TEST


START_TEST (test_QualExtension_getSBMLExtensionNamespaces)
{
  QualPkgNamespaces *qualns;
  qualns = static_cast<QualPkgNamespaces*>(G->getSBMLExtensionNamespaces(QUAL_XMLNS_L3V1V1));

  fail_unless(qualns->getLevel()          == 3);
  fail_unless(qualns->getVersion()        == 1);
  fail_unless(qualns->getPackageVersion() == 1);

  qualns = static_cast<QualPkgNamespaces*>(G->getSBMLExtensionNamespaces(""));

  fail_unless(qualns == NULL);
}
END_TEST


START_TEST(test_QualExtension_copy)
{
  QualExtension *g2 = new QualExtension(*G);

  fail_unless(g2->getName() == "qual");
  fail_unless(g2->getName() == QUAL_PACKAGE_NAME);

  fail_unless(g2->getURI(3,1,1) == QUAL_XMLNS_L3V1V1);
  fail_unless(g2->getURI(2,1,1) == "");
  fail_unless(g2->getURI(4,1,1) == "");

  fail_unless(g2->getLevel(QUAL_XMLNS_L3V1V1) == 3);
  fail_unless(g2->getLevel(CORE_XMLNS_L2V4)             == 0);
  fail_unless(g2->getLevel("")                          == 0);

  fail_unless(g2->getVersion(QUAL_XMLNS_L3V1V1) == 1);
  fail_unless(g2->getVersion(CORE_XMLNS_L2V4)             == 0);
  fail_unless(g2->getVersion("")                          == 0);

  fail_unless(g2->getPackageVersion(QUAL_XMLNS_L3V1V1) == 1);
  fail_unless(g2->getPackageVersion(CORE_XMLNS_L2V4)             == 0);
  fail_unless(g2->getPackageVersion("")                          == 0);

  delete g2;
}
END_TEST


START_TEST(test_QualExtension_assignment)
{
  QualExtension* g2 = new QualExtension();

  (*g2) = (*G);

  fail_unless(g2->getName() == "qual");
  fail_unless(g2->getName() == QUAL_PACKAGE_NAME);

  fail_unless(g2->getURI(3,1,1) == QUAL_XMLNS_L3V1V1);
  fail_unless(g2->getURI(2,1,1) == "");
  fail_unless(g2->getURI(4,1,1) == "");

  fail_unless(g2->getLevel(QUAL_XMLNS_L3V1V1) == 3);
  fail_unless(g2->getLevel(CORE_XMLNS_L2V4)             == 0);
  fail_unless(g2->getLevel("")                          == 0);

  fail_unless(g2->getVersion(QUAL_XMLNS_L3V1V1) == 1);
  fail_unless(g2->getVersion(CORE_XMLNS_L2V4)             == 0);
  fail_unless(g2->getVersion("")                          == 0);

  fail_unless(g2->getPackageVersion(QUAL_XMLNS_L3V1V1) == 1);
  fail_unless(g2->getPackageVersion(CORE_XMLNS_L2V4)             == 0);
  fail_unless(g2->getPackageVersion("")                          == 0);

  delete g2;
}
END_TEST


START_TEST(test_QualExtension_clone)
{
  QualExtension* g2 = G->clone();

  fail_unless(g2->getName() == "qual");
  fail_unless(g2->getName() == QUAL_PACKAGE_NAME);

  fail_unless(g2->getURI(3,1,1) == QUAL_XMLNS_L3V1V1);
  fail_unless(g2->getURI(2,1,1) == "");
  fail_unless(g2->getURI(4,1,1) == "");

  fail_unless(g2->getLevel(QUAL_XMLNS_L3V1V1) == 3);
  fail_unless(g2->getLevel(CORE_XMLNS_L2V4)             == 0);
  fail_unless(g2->getLevel("")                          == 0);

  fail_unless(g2->getVersion(QUAL_XMLNS_L3V1V1) == 1);
  fail_unless(g2->getVersion(CORE_XMLNS_L2V4)             == 0);
  fail_unless(g2->getVersion("")                          == 0);

  fail_unless(g2->getPackageVersion(QUAL_XMLNS_L3V1V1) == 1);
  fail_unless(g2->getPackageVersion(CORE_XMLNS_L2V4)             == 0);
  fail_unless(g2->getPackageVersion("")                          == 0);

  delete g2;
}
END_TEST


START_TEST(test_QualExtension_registry)
{
  const SBMLExtension* sbext = SBMLExtensionRegistry::getInstance().getExtension("qual");

  fail_unless(sbext != NULL);

  fail_unless(sbext->getName() == "qual");
  fail_unless(sbext->getName() == QUAL_PACKAGE_NAME);

  fail_unless(sbext->getURI(3,1,1) == QUAL_XMLNS_L3V1V1);
  fail_unless(sbext->getURI(2,1,1) == "");
  fail_unless(sbext->getURI(4,1,1) == "");

  fail_unless(sbext->getLevel(QUAL_XMLNS_L3V1V1) == 3);
  fail_unless(sbext->getLevel(CORE_XMLNS_L2V4)             == 0);
  fail_unless(sbext->getLevel("")                          == 0);

  fail_unless(sbext->getVersion(QUAL_XMLNS_L3V1V1) == 1);
  fail_unless(sbext->getVersion(CORE_XMLNS_L2V4)             == 0);
  fail_unless(sbext->getVersion("")                          == 0);

  fail_unless(sbext->getPackageVersion(QUAL_XMLNS_L3V1V1) == 1);
  fail_unless(sbext->getPackageVersion(CORE_XMLNS_L2V4)             == 0);
  fail_unless(sbext->getPackageVersion("")                          == 0);
}
END_TEST


START_TEST(test_QualExtension_typecode)
{
  const SBMLExtension* sbext = SBMLExtensionRegistry::getInstance().getExtension("qual");

  fail_unless(sbext != NULL);
  fail_unless(strcmp(sbext->getStringFromTypeCode(SBML_QUAL_QUALITATIVE_SPECIES), "QualitativeSpecies") == 0);
  fail_unless(strcmp(sbext->getStringFromTypeCode(SBML_QUAL_TRANSITION), "Transition") == 0);
  fail_unless(strcmp(sbext->getStringFromTypeCode(SBML_QUAL_INPUT), "Input") == 0);
  fail_unless(strcmp(sbext->getStringFromTypeCode(SBML_QUAL_OUTPUT), "Output") == 0);
  fail_unless(strcmp(sbext->getStringFromTypeCode(SBML_QUAL_FUNCTION_TERM), "FunctionTerm") == 0);
  fail_unless(strcmp(sbext->getStringFromTypeCode(SBML_QUAL_DEFAULT_TERM), "DefaultTerm") == 0);
  fail_unless(strcmp(sbext->getStringFromTypeCode(SBML_QUAL_QUALITATIVE_SPECIES-1), "(Unknown SBML Qual Type)") == 0);
  fail_unless(strcmp(sbext->getStringFromTypeCode(SBML_QUAL_DEFAULT_TERM+1), "(Unknown SBML Qual Type)") == 0);
}
END_TEST

START_TEST(test_QualExtension_SBMLtypecode)
{	
  fail_unless(strcmp(SBMLTypeCode_toString(SBML_QUAL_QUALITATIVE_SPECIES, "qual"), "QualitativeSpecies") == 0);
  fail_unless(strcmp(SBMLTypeCode_toString(SBML_QUAL_TRANSITION, "qual"), "Transition") == 0);
  fail_unless(strcmp(SBMLTypeCode_toString(SBML_QUAL_INPUT, "qual"), "Input") == 0);
  fail_unless(strcmp(SBMLTypeCode_toString(SBML_QUAL_OUTPUT, "qual"), "Output") == 0);
  fail_unless(strcmp(SBMLTypeCode_toString(SBML_QUAL_FUNCTION_TERM, "qual"), "FunctionTerm") == 0);
  fail_unless(strcmp(SBMLTypeCode_toString(SBML_QUAL_DEFAULT_TERM, "qual"), "DefaultTerm") == 0);
  fail_unless(strcmp(SBMLTypeCode_toString(SBML_QUAL_QUALITATIVE_SPECIES-1, "qual"), "(Unknown SBML Qual Type)") == 0);
  fail_unless(strcmp(SBMLTypeCode_toString(SBML_QUAL_DEFAULT_TERM+1, "qual"), "(Unknown SBML Qual Type)") == 0);
}
END_TEST

Suite *
create_suite_QualExtension (void)
{
  Suite *suite = suite_create("QualExtension");
  TCase *tcase = tcase_create("QualExtension");

  tcase_add_checked_fixture(tcase, QualExtensionTest_setup, QualExtensionTest_teardown);
 
  tcase_add_test( tcase, test_QualExtension_getName         );
  tcase_add_test( tcase, test_QualExtension_getURI          );
  tcase_add_test( tcase, test_QualExtension_getLevelVersion );
  tcase_add_test( tcase, test_QualExtension_getSBMLExtensionNamespaces);
  tcase_add_test( tcase, test_QualExtension_copy            );
  tcase_add_test( tcase, test_QualExtension_assignment      );
  tcase_add_test( tcase, test_QualExtension_clone           );
  tcase_add_test( tcase, test_QualExtension_registry        );
  tcase_add_test( tcase, test_QualExtension_typecode        );
  tcase_add_test( tcase, test_QualExtension_SBMLtypecode    );

  suite_add_tcase(suite, tcase);

  return suite;
}


CK_CPPEND
