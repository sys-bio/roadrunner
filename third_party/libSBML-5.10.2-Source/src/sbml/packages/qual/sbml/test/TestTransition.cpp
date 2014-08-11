/**
 * @file    TestTransition.cpp
 * @brief   TestTransition unit tests
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

static Transition* G; 
static QualPkgNamespaces* GNS;

void
TransitionTest_setup (void)
{
  GNS = new QualPkgNamespaces();
  G = new Transition(GNS);
  
  if (G == NULL)
  {
    fail("Failed to create a Transition object");
  }
}


void
TransitionTest_teardown (void)
{
  delete G;
  delete GNS;
}


START_TEST (test_Transition_create)
{
  fail_unless(G->isSetId()   == false);
  fail_unless(G->isSetName() == false);

  fail_unless(G->getNumInputs()        == 0);
  fail_unless(G->getNumOutputs()       == 0);
  fail_unless(G->getNumFunctionTerms() == 0);

}
END_TEST


START_TEST (test_Transition_id)
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


START_TEST (test_Transition_name)
{
  fail_unless(G->isSetName() == false);

  fail_unless(G->setName("i1") == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetName() == true);
  fail_unless(G->getName() == "i1");

  fail_unless(G->unsetName() == LIBSBML_OPERATION_SUCCESS);
  fail_unless(G->isSetName() == false);
}
END_TEST


START_TEST (test_Transition_Input)
{
  fail_unless(G->getNumInputs() == 0);

  G->createInput();

  fail_unless(G->getNumInputs() == 1);

  Input *i = G->createInput();
  i->setId("i1");

  fail_unless(G->getNumInputs() == 2);

  G->removeInput("i1");

  fail_unless(G->getNumInputs() == 1);

  G->removeInput(0);

  fail_unless(G->getNumInputs() == 0);
}
END_TEST


START_TEST (test_Transition_Output)
{
  fail_unless(G->getNumOutputs() == 0);

  G->createOutput();

  fail_unless(G->getNumOutputs() == 1);

  Output *i = G->createOutput();
  i->setId("i1");

  fail_unless(G->getNumOutputs() == 2);

  G->removeOutput("i1");

  fail_unless(G->getNumOutputs() == 1);

  G->removeOutput(0);

  fail_unless(G->getNumOutputs() == 0);
}
END_TEST


START_TEST (test_Transition_FunctionTerm)
{
  fail_unless(G->getNumFunctionTerms() == 0);

  G->createFunctionTerm();

  fail_unless(G->getNumFunctionTerms() == 1);

  //FunctionTerm *i =
  G->createFunctionTerm();

  fail_unless(G->getNumFunctionTerms() == 2);

  G->removeFunctionTerm(1);

  fail_unless(G->getNumFunctionTerms() == 1);

  G->removeFunctionTerm(0);

  fail_unless(G->getNumFunctionTerms() == 0);
}
END_TEST


START_TEST(test_Transition_copy)
{
  Output *o = G->createOutput();
  o->setQualitativeSpecies("s1");
  o->setTransitionEffect(OutputTransitionEffect_fromString("production"));

  G->setId("t1");

  Transition *g2 = new Transition(*G);

  fail_unless(g2->isSetName()           == false);
  fail_unless(g2->isSetId()             == true);
  fail_unless(g2->getId()               == "t1");
  fail_unless(g2->getNumInputs()        == 0);
  fail_unless(g2->getNumOutputs()       == 1);
  fail_unless(g2->getNumFunctionTerms() == 0);
  
  fail_unless(g2->getOutput(0)->isSetQualitativeSpecies() == true);
  fail_unless(g2->getOutput(0)->isSetId()                 == false);
  fail_unless(g2->getOutput(0)->getQualitativeSpecies()   == "s1");
  fail_unless(g2->getOutput(0)->getTransitionEffect()     == OUTPUT_TRANSITION_EFFECT_PRODUCTION);

 
  delete g2;
}
END_TEST


START_TEST(test_Transition_assignment)
{
  Output *o = G->createOutput();
  o->setQualitativeSpecies("s1");
  o->setTransitionEffect(OutputTransitionEffect_fromString("production"));

  G->setId("t1");

  Transition* g2 = new Transition();

  (*g2) = (*G);

  fail_unless(g2->isSetName()           == false);
  fail_unless(g2->isSetId()             == true);
  fail_unless(g2->getId()               == "t1");
  fail_unless(g2->getNumInputs()        == 0);
  fail_unless(g2->getNumOutputs()       == 1);
  fail_unless(g2->getNumFunctionTerms() == 0);
  
  fail_unless(g2->getOutput(0)->isSetQualitativeSpecies() == true);
  fail_unless(g2->getOutput(0)->isSetId()                 == false);
  fail_unless(g2->getOutput(0)->getQualitativeSpecies()   == "s1");
  fail_unless(g2->getOutput(0)->getTransitionEffect()     == OUTPUT_TRANSITION_EFFECT_PRODUCTION);

  delete g2;
}
END_TEST


START_TEST(test_Transition_clone)
{
  Output *o = G->createOutput();
  o->setQualitativeSpecies("s1");
  o->setTransitionEffect(OutputTransitionEffect_fromString("production"));

  G->setId("t1");

  Transition* g2 = G->clone();
  
  fail_unless(g2->isSetName()           == false);
  fail_unless(g2->isSetId()             == true);
  fail_unless(g2->getId()               == "t1");
  fail_unless(g2->getNumInputs()        == 0);
  fail_unless(g2->getNumOutputs()       == 1);
  fail_unless(g2->getNumFunctionTerms() == 0);
  
  fail_unless(g2->getOutput(0)->isSetQualitativeSpecies() == true);
  fail_unless(g2->getOutput(0)->isSetId()                 == false);
  fail_unless(g2->getOutput(0)->getQualitativeSpecies()   == "s1");
  fail_unless(g2->getOutput(0)->getTransitionEffect()     == OUTPUT_TRANSITION_EFFECT_PRODUCTION);
  
  delete g2;
}
END_TEST


Suite *
create_suite_Transition (void)
{
  Suite *suite = suite_create("Transition");
  TCase *tcase = tcase_create("Transition");

  tcase_add_checked_fixture(tcase, TransitionTest_setup, TransitionTest_teardown);
 
  tcase_add_test( tcase, test_Transition_create       );

  tcase_add_test( tcase, test_Transition_id           );
  tcase_add_test( tcase, test_Transition_name         );
  tcase_add_test( tcase, test_Transition_Input        );
  tcase_add_test( tcase, test_Transition_FunctionTerm );
  tcase_add_test( tcase, test_Transition_Output       );

  tcase_add_test( tcase, test_Transition_copy            );
  tcase_add_test( tcase, test_Transition_assignment      );
  tcase_add_test( tcase, test_Transition_clone           );

  suite_add_tcase(suite, tcase);

  return suite;
}


CK_CPPEND
