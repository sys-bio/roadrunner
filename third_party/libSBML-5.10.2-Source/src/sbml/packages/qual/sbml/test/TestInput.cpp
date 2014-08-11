/**
 * @file    TestInput.cpp
 * @brief   TestInput unit tests
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

static Input* G; 
static QualPkgNamespaces* GNS;

void
InputTest_setup (void)
{
  GNS = new QualPkgNamespaces();
  G = new Input(GNS);
  
  if (G == NULL)
  {
    fail("Failed to create a Input object");
  }
}


void
InputTest_teardown (void)
{
  delete G;
  delete GNS;
}


START_TEST (test_Input_create)
{
  fail_unless(G->isSetId()                 == false);
  fail_unless(G->isSetName()               == false);
  fail_unless(G->isSetQualitativeSpecies() == false);
  fail_unless(G->isSetSign()               == false);
  fail_unless(G->isSetTransitionEffect()   == false);
  fail_unless(G->isSetThresholdLevel()     == false);

}
END_TEST


START_TEST (test_Input_id)
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


START_TEST (test_Input_name)
{
  fail_unless(G->isSetName() == false);

  fail_unless(G->setName("i1") == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetName() == true);
  fail_unless(G->getName() == "i1");

  fail_unless(G->unsetName() == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetName() == false);
}
END_TEST


START_TEST (test_Input_qualitativeSpecies)
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


START_TEST (test_Input_sign)
{
  fail_unless(G->isSetSign() == false);

  fail_unless(G->setSign(InputSign_fromString("p")) == LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(G->isSetSign() == false);

  fail_unless(G->setSign(InputSign_fromString("positive")) == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetSign() == true);
  fail_unless(!strcmp(InputSign_toString(G->getSign()), "positive"));

  fail_unless(G->setSign(INPUT_SIGN_NEGATIVE) == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetSign() == true);
  fail_unless(!strcmp(InputSign_toString(G->getSign()), "negative"));

  fail_unless(G->unsetSign() == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetSign() == false);
}
END_TEST


START_TEST (test_Input_transitionEffect)
{
  fail_unless(G->isSetTransitionEffect() == false);

  fail_unless(G->setTransitionEffect(
    InputTransitionEffect_fromString("o")) == LIBSBML_INVALID_ATTRIBUTE_VALUE);
  fail_unless(G->isSetTransitionEffect() == false);

  fail_unless(G->setTransitionEffect(
    InputTransitionEffect_fromString("none")) == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetTransitionEffect() == true);
  fail_unless(G->getTransitionEffect() == INPUT_TRANSITION_EFFECT_NONE);

  fail_unless(G->unsetTransitionEffect() == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetTransitionEffect() == false);
}
END_TEST


START_TEST (test_Input_thresholdLevel)
{
  fail_unless(G->isSetThresholdLevel() == false);

  fail_unless(G->setThresholdLevel(1) == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetThresholdLevel() == true);
  fail_unless(G->getThresholdLevel() == 1);

  fail_unless(G->unsetThresholdLevel() == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetThresholdLevel() == false);
}
END_TEST


START_TEST(test_Input_copy)
{
  G->setQualitativeSpecies("s1");
  G->setTransitionEffect(InputTransitionEffect_fromString("none"));

  Input *g2 = new Input(*G);

  fail_unless(g2->isSetName()               == false);
  fail_unless(g2->isSetId()                 == false);
  fail_unless(g2->isSetQualitativeSpecies() == true);
  fail_unless(g2->getQualitativeSpecies()   == "s1");
  fail_unless(g2->isSetSign()               == false);
  fail_unless(g2->isSetTransitionEffect()   == true);
  fail_unless(!strcmp(
    InputTransitionEffect_toString(g2->getTransitionEffect()), "none"));
  fail_unless(g2->isSetThresholdLevel()     == false);
 
  delete g2;
}
END_TEST


START_TEST(test_Input_assignment)
{
  G->setQualitativeSpecies("s1");
  G->setTransitionEffect(INPUT_TRANSITION_EFFECT_NONE);
  Input* g2 = new Input();

  (*g2) = (*G);

  fail_unless(g2->isSetName()               == false);
  fail_unless(g2->isSetId()                 == false);
  fail_unless(g2->isSetQualitativeSpecies() == true);
  fail_unless(g2->getQualitativeSpecies()   == "s1");
  fail_unless(g2->isSetSign()               == false);
  fail_unless(g2->isSetTransitionEffect()   == true);
  fail_unless(g2->getTransitionEffect()     == INPUT_TRANSITION_EFFECT_NONE);
  fail_unless(g2->isSetThresholdLevel()     == false);

  delete g2;
}
END_TEST


START_TEST(test_Input_clone)
{
  G->setQualitativeSpecies("s1");
  G->setTransitionEffect(INPUT_TRANSITION_EFFECT_NONE);
  Input* g2 = G->clone();
  
  fail_unless(g2->isSetName()               == false);
  fail_unless(g2->isSetId()                 == false);
  fail_unless(g2->isSetQualitativeSpecies() == true);
  fail_unless(g2->getQualitativeSpecies()   == "s1");
  fail_unless(g2->isSetSign()               == false);
  fail_unless(g2->isSetTransitionEffect()   == true);
  fail_unless(g2->getTransitionEffect()     == INPUT_TRANSITION_EFFECT_NONE);
  fail_unless(g2->isSetThresholdLevel()     == false);
  
  delete g2;
}
END_TEST


Suite *
create_suite_Input (void)
{
  Suite *suite = suite_create("Input");
  TCase *tcase = tcase_create("Input");

  tcase_add_checked_fixture(tcase, InputTest_setup, InputTest_teardown);
 
  tcase_add_test( tcase, test_Input_create         );

  tcase_add_test( tcase, test_Input_id                 );
  tcase_add_test( tcase, test_Input_name               );
  tcase_add_test( tcase, test_Input_qualitativeSpecies );
  tcase_add_test( tcase, test_Input_sign               );
  tcase_add_test( tcase, test_Input_thresholdLevel     );
  tcase_add_test( tcase, test_Input_transitionEffect   );

  tcase_add_test( tcase, test_Input_copy            );
  tcase_add_test( tcase, test_Input_assignment      );
  tcase_add_test( tcase, test_Input_clone           );

  suite_add_tcase(suite, tcase);

  return suite;
}


CK_CPPEND
