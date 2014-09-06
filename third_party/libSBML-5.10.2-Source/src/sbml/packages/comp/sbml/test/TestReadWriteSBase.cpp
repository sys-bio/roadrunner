/**
 * \file    TestSBaseObjects.cpp
 * \brief   Implementation of the Tests for the Comp sbase objects
 * \author  Lucian Smith
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
 * Copyright 2011-2012 jointly by the following organizations:
 *     1. California Institute of Technology, Pasadena, CA, USA
 *     2. EMBL European Bioinformatics Institute (EMBL-EBI), Hinxton, UK
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation.  A copy of the license agreement is provided
 * in the file named "LICENSE.txt" included with this software distribution
 * and also available online as http://sbml.org/software/libsbml/license.html
 * ---------------------------------------------------------------------- -->*/

 #include <sbml/common/common.h>

#include <sbml/packages/comp/common/CompExtensionTypes.h>

#include <sbml/SBMLReader.h>
#include <sbml/SBMLTypes.h>

#include <string>

#include <check.h>

using namespace std;

LIBSBML_CPP_NAMESPACE_USE

BEGIN_C_DECLS


extern char *TestDataDirectory;

START_TEST (test_comp_rwports)
{
  //Create a model with a port.
  SBMLNamespaces sbmlns(3,1,"comp",1);
  CompPkgNamespaces csbmlns(3,1,1,"comp");

  SBMLDocument orig(&sbmlns);
  CompSBMLDocumentPlugin* compdoc = static_cast<CompSBMLDocumentPlugin*>(orig.getPlugin("comp"));
  compdoc->setRequired(true);

  Model* model = orig.createModel();
  CompModelPlugin* compmod = static_cast<CompModelPlugin*>(model->getPlugin("comp"));

  Parameter* param = model->createParameter();
  param->setId("p1");
  param->setConstant(true);

  Port* port = compmod->createPort();
  port->setId("port1");
  port->setIdRef("p1");
  port->setSBOTerm(10);

  string original = writeSBMLToString(&orig);

  // Now round-trip that model.
  SBMLDocument* doc = readSBMLFromString(original.c_str());
  string rtmodel = writeSBMLToString(doc);

  fail_unless(original == rtmodel);
  delete doc;
}
END_TEST

  
Suite *
create_suite_TestReadWriteSBaseObjects(void)
{ 
  TCase *tcase = tcase_create("SBMLCompSBaseObjects");
  Suite *suite = suite_create("SBMLCompSBaseObjects");
  
  tcase_add_test(tcase, test_comp_rwports);

  suite_add_tcase(suite, tcase);

  return suite;
}


END_C_DECLS

