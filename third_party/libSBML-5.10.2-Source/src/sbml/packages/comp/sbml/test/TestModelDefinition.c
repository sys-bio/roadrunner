/**
 * \file    TestModelDefinition.c
 * \brief   ModelDefinition unit tests
 * \author  Sarah Keating
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

#include <sbml/packages/comp/sbml/ModelDefinition.h>
#include <sbml/packages/comp/extension/CompExtension.h>

#include <sbml/SBMLDocument.h>

#include <check.h>



#include <sbml/common/extern.h>

LIBSBML_CPP_NAMESPACE_USE


BEGIN_C_DECLS

static ModelDefinition_t *P;


void
ModelDefinitionTest_setup (void)
{
  P = ModelDefinition_create(3, 1, 1);

  if (P == NULL)
  {
    fail("ModelDefinition_create(3, 1, 1) returned a NULL pointer.");
  }
}


void
ModelDefinitionTest_teardown (void)
{
  ModelDefinition_free(P);
}


START_TEST (test_comp_modelDefinition_create)
{
  fail_unless( SBase_getTypeCode  ((SBase_t *) P) == SBML_COMP_MODELDEFINITION );
  fail_unless( SBase_getMetaId    ((SBase_t *) P) == NULL );
  fail_unless( SBase_getNotes     ((SBase_t *) P) == NULL );
  fail_unless( SBase_getAnnotation((SBase_t *) P) == NULL );

}
END_TEST


START_TEST (test_comp_modelDefinition_free_NULL)
{
  ModelDefinition_free(NULL);
}
END_TEST


START_TEST (test_comp_modelDefinition_id)
{
  const char *id = "mitochondria";


  fail_unless( !Model_isSetId(P) );
  
  Model_setId(P, id);

  fail_unless( !strcmp(Model_getId(P), id) );
  fail_unless( Model_isSetId(P) );

  if (Model_getId(P) == id)
  {
    fail("Model_setId(...) did not make a copy of string.");
  }
 
  Model_unsetId(P);
  
  fail_unless( !Model_isSetId(P) );

  if (Model_getId(P) != NULL)
  {
    fail("Model_unsetId(P) did not clear string.");
  }
}
END_TEST


START_TEST (test_comp_modelDefinition_name)
{
  const char *name = "My_Favorite_Factory";


  fail_unless( !Model_isSetName(P) );

  Model_setName(P, name);

  fail_unless( !strcmp(Model_getName(P), name) );
  fail_unless( Model_isSetName(P) );

  if (Model_getName(P) == name)
  {
    fail("Model_setName(...) did not make a copy of string.");
  }

  Model_unsetName(P);
  
  fail_unless( !Model_isSetName(P) );

  if (Model_getName(P) != NULL)
  {
    fail("Model_unsetName(P) did not clear string.");
  }
}
END_TEST


Suite *
create_suite_TestComp_ModelDefinition (void)
{
  Suite *suite = suite_create("comp_modelDefinition");
  TCase *tcase = tcase_create("comp_modelDefinition");


  tcase_add_checked_fixture( tcase,
                             ModelDefinitionTest_setup,
                             ModelDefinitionTest_teardown );

  tcase_add_test( tcase, test_comp_modelDefinition_create              );
  tcase_add_test( tcase, test_comp_modelDefinition_free_NULL           );
  tcase_add_test( tcase, test_comp_modelDefinition_id               );
  tcase_add_test( tcase, test_comp_modelDefinition_name             );

  suite_add_tcase(suite, tcase);

  return suite;
}

END_C_DECLS

