/**
 * @file    TestQualitativeSpecies.cpp
 * @brief   TestQualitativeSpecies unit tests
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

static QualitativeSpecies* G; 
static QualPkgNamespaces* GNS;

void
QualitativeSpeciesTest_setup (void)
{
  GNS = new QualPkgNamespaces();
  G = new QualitativeSpecies(GNS);
  
  if (G == NULL)
  {
    fail("Failed to create a QualitativeSpecies object");
  }
}


void
QualitativeSpeciesTest_teardown (void)
{
  delete G;
  delete GNS;
}


START_TEST (test_QualitativeSpecies_create)
{
  fail_unless(G->isSetId()               == false);
  fail_unless(G->isSetName()             == false);
  fail_unless(G->isSetCompartment()      == false);
  fail_unless(G->isSetConstant()         == false);
  fail_unless(G->isSetInitialLevel()     == false);
  fail_unless(G->isSetMaxLevel()         == false);

}
END_TEST


START_TEST (test_QualitativeSpecies_id)
{
  fail_unless(G->isSetId() == false);

  fail_unless(G->setId("1") == LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(G->isSetId() == false);

  fail_unless(G->setId("i1") == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetId() == true);
  fail_unless(G->getId() == "i1");

  fail_unless(G->unsetId() == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetId() == false);
}
END_TEST


START_TEST (test_QualitativeSpecies_name)
{
  fail_unless(G->isSetName() == false);

  fail_unless(G->setName("i1") == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetName() == true);
  fail_unless(G->getName() == "i1");

  fail_unless(G->unsetName() == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetName() == false);
}
END_TEST


START_TEST (test_QualitativeSpecies_compartment)
{
  fail_unless(G->isSetCompartment() == false);

  fail_unless(G->setCompartment("1") == LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(G->isSetCompartment() == false);

  fail_unless(G->setCompartment("i1") == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetCompartment() == true);
  fail_unless(G->getCompartment() == "i1");

  fail_unless(G->unsetCompartment() == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetCompartment() == false);
}
END_TEST


START_TEST (test_QualitativeSpecies_constant)
{
  fail_unless(G->isSetConstant() == false);

  fail_unless(G->setConstant(true) == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetConstant() == true);
  fail_unless(G->getConstant() == true);

  fail_unless(G->setConstant(false) == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetConstant() == true);
  fail_unless(G->getConstant() == false);

  fail_unless(G->unsetConstant() == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetConstant() == false);
}
END_TEST


START_TEST (test_QualitativeSpecies_initialLevel)
{
  fail_unless(G->isSetInitialLevel() == false);

  fail_unless(G->setInitialLevel(1) == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetInitialLevel() == true);
  fail_unless(G->getInitialLevel() == 1);

  fail_unless(G->unsetInitialLevel() == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetInitialLevel() == false);
}
END_TEST


START_TEST (test_QualitativeSpecies_maxLevel)
{
  fail_unless(G->isSetMaxLevel() == false);

  fail_unless(G->setMaxLevel(1) == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetMaxLevel() == true);
  fail_unless(G->getMaxLevel() == 1);

  fail_unless(G->unsetMaxLevel() == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetMaxLevel() == false);
}
END_TEST


START_TEST(test_QualitativeSpecies_copy)
{
  G->setId("s1");
  G->setCompartment("c");
  G->setConstant(true);
  G->setInitialLevel(2);
  G->setMaxLevel(4);


  QualitativeSpecies *g2 = new QualitativeSpecies(*G);

  fail_unless(g2->isSetName()         == false);
  fail_unless(g2->isSetId()           == true);
  fail_unless(g2->getId()             == "s1");
  fail_unless(g2->isSetConstant()     == true);
  fail_unless(g2->getConstant()       == true);
  fail_unless(g2->isSetCompartment()  == true);
  fail_unless(g2->getCompartment()    == "c");
  fail_unless(g2->isSetInitialLevel() == true);
  fail_unless(g2->getInitialLevel()   == 2);
  fail_unless(g2->isSetMaxLevel()     == true);
  fail_unless(g2->getMaxLevel()       == 4);
 
  delete g2;
}
END_TEST


START_TEST(test_QualitativeSpecies_assignment)
{
  G->setId("s1");
  G->setCompartment("c");
  G->setConstant(true);
  G->setInitialLevel(2);
  G->setMaxLevel(4);

  QualitativeSpecies* g2 = new QualitativeSpecies();

  (*g2) = (*G);

  fail_unless(g2->isSetName()         == false);
  fail_unless(g2->isSetId()           == true);
  fail_unless(g2->getId()             == "s1");
  fail_unless(g2->isSetConstant()     == true);
  fail_unless(g2->getConstant()       == true);
  fail_unless(g2->isSetCompartment()  == true);
  fail_unless(g2->getCompartment()    == "c");
  fail_unless(g2->isSetInitialLevel() == true);
  fail_unless(g2->getInitialLevel()   == 2);
  fail_unless(g2->isSetMaxLevel()     == true);
  fail_unless(g2->getMaxLevel()       == 4);
 
  delete g2;
}
END_TEST


START_TEST(test_QualitativeSpecies_clone)
{
  G->setId("s1");
  G->setCompartment("c");
  G->setConstant(true);
  G->setInitialLevel(2);
  G->setMaxLevel(4);
  
  QualitativeSpecies* g2 = G->clone();
  
  fail_unless(g2->isSetName()         == false);
  fail_unless(g2->isSetId()           == true);
  fail_unless(g2->getId()             == "s1");
  fail_unless(g2->isSetConstant()     == true);
  fail_unless(g2->getConstant()       == true);
  fail_unless(g2->isSetCompartment()  == true);
  fail_unless(g2->getCompartment()    == "c");
  fail_unless(g2->isSetInitialLevel() == true);
  fail_unless(g2->getInitialLevel()   == 2);
  fail_unless(g2->isSetMaxLevel()     == true);
  fail_unless(g2->getMaxLevel()       == 4);
 
  delete g2;
}
END_TEST


Suite *
create_suite_QualitativeSpecies (void)
{
  Suite *suite = suite_create("QualitativeSpecies");
  TCase *tcase = tcase_create("QualitativeSpecies");

  tcase_add_checked_fixture(tcase, QualitativeSpeciesTest_setup, QualitativeSpeciesTest_teardown);
 
  tcase_add_test( tcase, test_QualitativeSpecies_create         );

  tcase_add_test( tcase, test_QualitativeSpecies_id           );
  tcase_add_test( tcase, test_QualitativeSpecies_name         );
  tcase_add_test( tcase, test_QualitativeSpecies_compartment  );
  tcase_add_test( tcase, test_QualitativeSpecies_constant     );
  tcase_add_test( tcase, test_QualitativeSpecies_initialLevel );
  tcase_add_test( tcase, test_QualitativeSpecies_maxLevel     );

  tcase_add_test( tcase, test_QualitativeSpecies_copy            );
  tcase_add_test( tcase, test_QualitativeSpecies_assignment      );
  tcase_add_test( tcase, test_QualitativeSpecies_clone           );

  suite_add_tcase(suite, tcase);

  return suite;
}


CK_CPPEND
