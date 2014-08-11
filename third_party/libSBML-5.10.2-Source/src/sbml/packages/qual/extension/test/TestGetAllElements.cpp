/**
 * @file    TestGetAllElements.cpp
 * @brief   Unit tests of writing QualExtension 
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
#include <sbml/SBMLTypes.h>
#include <sbml/common/extern.h>
#include <sbml/extension/SBMLExtensionRegistry.h>
#include <sbml/packages/qual/common/QualExtensionTypes.h>
#include <string>

/** @cond doxygenIgnored */

using namespace std;
LIBSBML_CPP_NAMESPACE_USE

/** @endcond doxygenIgnored */


CK_CPPSTART

extern char *TestDataDirectory;

START_TEST (test_getAllElements_transition)
{
  char *filename = safe_strcat(TestDataDirectory, "qual-example1.xml");
  SBMLDocument *document = readSBMLFromFile(filename);
  
  Model *model = document->getModel();

  fail_unless(model != NULL);

  QualModelPlugin* mplugin = static_cast<QualModelPlugin*>(model->getPlugin("qual"));
  fail_unless(mplugin != NULL);

  Transition* t = mplugin->getTransition(0);

  
  fail_unless(t->getPackageName() == "qual");
  fail_unless(t->getId() == "d");
  fail_unless(t->getNumInputs() == 1);
  fail_unless(t->getNumOutputs() == 1);
  fail_unless(t->getNumFunctionTerms() == 1);
  fail_unless(t->isSetDefaultTerm() == true);

  List * allElems = t->getAllElements();

  fail_unless(allElems->getSize() == 7);

  fail_unless ( allElems->get(0) == t->getListOfInputs());
  fail_unless ( allElems->get(1) == t->getInput(0));
  fail_unless ( allElems->get(2) == t->getListOfOutputs());
  fail_unless ( allElems->get(3) == t->getOutput(0));
  fail_unless ( allElems->get(4) == t->getListOfFunctionTerms());
  fail_unless ( allElems->get(5) == t->getFunctionTerm(0));
  fail_unless ( allElems->get(6) == t->getDefaultTerm());

  delete allElems;
  delete document;  
}
END_TEST


Suite *
create_suite_GetAllElements (void)
{
  Suite *suite = suite_create("GetAllElements");
  TCase *tcase = tcase_create("GetAllElements");

  tcase_add_test( tcase, test_getAllElements_transition);
  suite_add_tcase(suite, tcase);

  return suite;
}


CK_CPPEND
