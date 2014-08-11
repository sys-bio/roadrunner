/**
 * @file    TestOutput.cpp
 * @brief   TestOutput unit tests
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

static Output* G; 
static QualPkgNamespaces* GNS;

void
OutputTest_setup (void)
{
  GNS = new QualPkgNamespaces();
  G = new Output(GNS);
  
  if (G == NULL)
  {
    fail("Failed to create a Output object");
  }
}


void
OutputTest_teardown (void)
{
  delete G;
  delete GNS;
}


START_TEST (test_Output_create)
{
  fail_unless(G->isSetId()                 == false);
  fail_unless(G->isSetName()               == false);
  fail_unless(G->isSetQualitativeSpecies() == false);
  fail_unless(G->isSetTransitionEffect()   == false);
  fail_unless(G->isSetOutputLevel()        == false);

}
END_TEST


START_TEST (test_Output_id)
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


START_TEST (test_Output_name)
{
  fail_unless(G->isSetName() == false);

  fail_unless(G->setName("i1") == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetName() == true);
  fail_unless(G->getName() == "i1");

  fail_unless(G->unsetName() == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetName() == false);
}
END_TEST


START_TEST (test_Output_qualitativeSpecies)
{
  fail_unless(G->isSetQualitativeSpecies() == false);

  fail_unless(G->setQualitativeSpecies("1") == LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(G->isSetQualitativeSpecies() == false);

  fail_unless(G->setQualitativeSpecies("i1") == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetQualitativeSpecies() == true);
  fail_unless(G->getQualitativeSpecies() == "i1");

  fail_unless(G->unsetQualitativeSpecies() == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetQualitativeSpecies() == false);
}
END_TEST


START_TEST (test_Output_transitionEffect)
{
  fail_unless(G->isSetTransitionEffect() == false);

  fail_unless(G->setTransitionEffect(
    OutputTransitionEffect_fromString("o")) == LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(G->isSetTransitionEffect() == false);

  fail_unless(G->setTransitionEffect(
    OutputTransitionEffect_fromString("production")) == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetTransitionEffect() == true);
  fail_unless(G->getTransitionEffect() == OUTPUT_TRANSITION_EFFECT_PRODUCTION);

  fail_unless(G->unsetTransitionEffect() == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetTransitionEffect() == false);
}
END_TEST


START_TEST (test_Output_OutputLevel)
{
  fail_unless(G->isSetOutputLevel() == false);

  fail_unless(G->setOutputLevel(1) == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetOutputLevel() == true);
  fail_unless(G->getOutputLevel() == 1);

  fail_unless(G->unsetOutputLevel() == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetOutputLevel() == false);
}
END_TEST


START_TEST(test_Output_copy)
{
  G->setQualitativeSpecies("s1");
  G->setTransitionEffect(OutputTransitionEffect_fromString("production"));

  Output *g2 = new Output(*G);

  fail_unless(g2->isSetName()               == false);
  fail_unless(g2->isSetId()                 == false);
  fail_unless(g2->isSetQualitativeSpecies() == true);
  fail_unless(g2->getQualitativeSpecies()   == "s1");
  fail_unless(g2->isSetTransitionEffect()   == true);
  fail_unless(!strcmp(
    OutputTransitionEffect_toString(g2->getTransitionEffect()), "production"));
  fail_unless(g2->isSetOutputLevel()        == false);
 
  delete g2;
}
END_TEST


START_TEST(test_Output_assignment)
{
  G->setQualitativeSpecies("s1");
  G->setTransitionEffect(OUTPUT_TRANSITION_EFFECT_PRODUCTION);
  Output* g2 = new Output();

  (*g2) = (*G);

  fail_unless(g2->isSetName()               == false);
  fail_unless(g2->isSetId()                 == false);
  fail_unless(g2->isSetQualitativeSpecies() == true);
  fail_unless(g2->getQualitativeSpecies()   == "s1");
  fail_unless(g2->isSetTransitionEffect()   == true);
  fail_unless(g2->getTransitionEffect()     == OUTPUT_TRANSITION_EFFECT_PRODUCTION);
  fail_unless(g2->isSetOutputLevel()        == false);

  delete g2;
}
END_TEST


START_TEST(test_Output_clone)
{
  G->setQualitativeSpecies("s1");
  G->setTransitionEffect(OUTPUT_TRANSITION_EFFECT_PRODUCTION);
  Output* g2 = G->clone();
  
  fail_unless(g2->isSetName()               == false);
  fail_unless(g2->isSetId()                 == false);
  fail_unless(g2->isSetQualitativeSpecies() == true);
  fail_unless(g2->getQualitativeSpecies()   == "s1");
  fail_unless(g2->isSetTransitionEffect()   == true);
  fail_unless(g2->getTransitionEffect()     == OUTPUT_TRANSITION_EFFECT_PRODUCTION);
  fail_unless(g2->isSetOutputLevel()        == false);
  
  delete g2;
}
END_TEST


Suite *
create_suite_Output (void)
{
  Suite *suite = suite_create("Output");
  TCase *tcase = tcase_create("Output");

  tcase_add_checked_fixture(tcase, OutputTest_setup, OutputTest_teardown);
 
  tcase_add_test( tcase, test_Output_create         );

  tcase_add_test( tcase, test_Output_id                  );
  tcase_add_test( tcase, test_Output_name                );
  tcase_add_test( tcase, test_Output_qualitativeSpecies  );
  tcase_add_test( tcase, test_Output_OutputLevel         );
  tcase_add_test( tcase, test_Output_transitionEffect    );

  tcase_add_test( tcase, test_Output_copy            );
  tcase_add_test( tcase, test_Output_assignment      );
  tcase_add_test( tcase, test_Output_clone           );

  suite_add_tcase(suite, tcase);

  return suite;
}


CK_CPPEND
