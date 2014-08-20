/**
 * \file    TestExternalModelDefinition.c
 * \brief   ExternalModelDefinition unit tests
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

#include <sbml/packages/comp/sbml/ExternalModelDefinition.h>
#include <sbml/packages/comp/extension/CompExtension.h>

#include <sbml/SBMLDocument.h>

#include <check.h>



#include <sbml/common/extern.h>

LIBSBML_CPP_NAMESPACE_USE


BEGIN_C_DECLS

static ExternalModelDefinition_t *P;


void
ExternalModelDefinitionTest_setup (void)
{
  P = ExternalModelDefinition_create(3, 1, 1);

  if (P == NULL)
  {
    fail("ExternalModelDefinition_create(3, 1, 1) returned a NULL pointer.");
  }
}


void
ExternalModelDefinitionTest_teardown (void)
{
  ExternalModelDefinition_free(P);
}


START_TEST (test_comp_externalModelDefinition_create)
{
  fail_unless( SBase_getTypeCode  ((SBase_t *) P) == SBML_COMP_EXTERNALMODELDEFINITION );
  fail_unless( SBase_getMetaId    ((SBase_t *) P) == NULL );
  fail_unless( SBase_getNotes     ((SBase_t *) P) == NULL );
  fail_unless( SBase_getAnnotation((SBase_t *) P) == NULL );

  fail_unless( ExternalModelDefinition_getId     (P) == NULL );
  fail_unless( ExternalModelDefinition_getName   (P) == NULL );
  fail_unless( ExternalModelDefinition_getModelRef   (P) == NULL );
  fail_unless( ExternalModelDefinition_getSource   (P) == NULL );

  fail_unless( !ExternalModelDefinition_isSetId     (P) );
  fail_unless( !ExternalModelDefinition_isSetName   (P) );
  fail_unless( !ExternalModelDefinition_isSetModelRef   (P) );
  fail_unless( !ExternalModelDefinition_isSetSource   (P) );
}
END_TEST


START_TEST (test_comp_externalModelDefinition_free_NULL)
{
  ExternalModelDefinition_free(NULL);
}
END_TEST


START_TEST (test_comp_externalModelDefinition_id)
{
  const char *id = "mitochondria";


  fail_unless( !ExternalModelDefinition_isSetId(P) );
  
  ExternalModelDefinition_setId(P, id);

  fail_unless( !strcmp(ExternalModelDefinition_getId(P), id) );
  fail_unless( ExternalModelDefinition_isSetId(P) );

  if (ExternalModelDefinition_getId(P) == id)
  {
    fail("ExternalModelDefinition_setId(...) did not make a copy of string.");
  }
 
  ExternalModelDefinition_unsetId(P);
  
  fail_unless( !ExternalModelDefinition_isSetId(P) );

  if (ExternalModelDefinition_getId(P) != NULL)
  {
    fail("ExternalModelDefinition_unsetId(P) did not clear string.");
  }
}
END_TEST


START_TEST (test_comp_externalModelDefinition_name)
{
  const char *name = "My_Favorite_Factory";


  fail_unless( !ExternalModelDefinition_isSetName(P) );

  ExternalModelDefinition_setName(P, name);

  fail_unless( !strcmp(ExternalModelDefinition_getName(P), name) );
  fail_unless( ExternalModelDefinition_isSetName(P) );

  if (ExternalModelDefinition_getName(P) == name)
  {
    fail("ExternalModelDefinition_setName(...) did not make a copy of string.");
  }

  ExternalModelDefinition_unsetName(P);
  
  fail_unless( !ExternalModelDefinition_isSetName(P) );

  if (ExternalModelDefinition_getName(P) != NULL)
  {
    fail("ExternalModelDefinition_unsetName(P) did not clear string.");
  }
}
END_TEST


START_TEST (test_comp_externalModelDefinition_modelRef)
{
  const char *modelRef = "My_Favorite_Factory";


  fail_unless( !ExternalModelDefinition_isSetModelRef(P) );

  ExternalModelDefinition_setModelRef(P, modelRef);

  fail_unless( !strcmp(ExternalModelDefinition_getModelRef(P), modelRef) );
  fail_unless( ExternalModelDefinition_isSetModelRef(P) );

  if (ExternalModelDefinition_getModelRef(P) == modelRef)
  {
    fail("ExternalModelDefinition_setModelRef(...) did not make a copy of string.");
  }

  ExternalModelDefinition_unsetModelRef(P);
  
  fail_unless( !ExternalModelDefinition_isSetModelRef(P) );

  if (ExternalModelDefinition_getModelRef(P) != NULL)
  {
    fail("ExternalModelDefinition_unsetModelRef(P) did not clear string.");
  }
}
END_TEST


START_TEST (test_comp_externalModelDefinition_source)
{
  const char *source = "My_Favorite_Factory";


  fail_unless( !ExternalModelDefinition_isSetSource(P) );

  ExternalModelDefinition_setSource(P, source);

  fail_unless( !strcmp(ExternalModelDefinition_getSource(P), source) );
  fail_unless( ExternalModelDefinition_isSetSource(P) );

  if (ExternalModelDefinition_getSource(P) == source)
  {
    fail("ExternalModelDefinition_setSource(...) did not make a copy of string.");
  }

  ExternalModelDefinition_unsetSource(P);
  
  fail_unless( !ExternalModelDefinition_isSetSource(P) );

  if (ExternalModelDefinition_getSource(P) != NULL)
  {
    fail("ExternalModelDefinition_unsetSource(P) did not clear string.");
  }
}
END_TEST


START_TEST (test_comp_externalModelDefinition_hasRequiredAttributes )
{
  ExternalModelDefinition_t *P = ExternalModelDefinition_create (3, 1, 1);

  fail_unless ( !ExternalModelDefinition_hasRequiredAttributes(P));

  ExternalModelDefinition_setId(P, "p");

  fail_unless ( !ExternalModelDefinition_hasRequiredAttributes(P));

  ExternalModelDefinition_setSource(P, "http://here.com/it/is");
  
  fail_unless ( ExternalModelDefinition_hasRequiredAttributes(P));

  ExternalModelDefinition_free(P);
}
END_TEST


Suite *
create_suite_TestComp_ExternalModelDefinition (void)
{
  Suite *suite = suite_create("comp_externalModelDefinition");
  TCase *tcase = tcase_create("comp_externalModelDefinition");


  tcase_add_checked_fixture( tcase,
                             ExternalModelDefinitionTest_setup,
                             ExternalModelDefinitionTest_teardown );

  tcase_add_test( tcase, test_comp_externalModelDefinition_create              );
  tcase_add_test( tcase, test_comp_externalModelDefinition_free_NULL           );
  tcase_add_test( tcase, test_comp_externalModelDefinition_id               );
  tcase_add_test( tcase, test_comp_externalModelDefinition_name             );
  tcase_add_test( tcase, test_comp_externalModelDefinition_modelRef         );
  tcase_add_test( tcase, test_comp_externalModelDefinition_source           );
  tcase_add_test( tcase, test_comp_externalModelDefinition_hasRequiredAttributes        );

  suite_add_tcase(suite, tcase);

  return suite;
}

END_C_DECLS

