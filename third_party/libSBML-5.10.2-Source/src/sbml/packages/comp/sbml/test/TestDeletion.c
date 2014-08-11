/**
 * \file    TestDeletion.c
 * \brief   Deletion unit tests
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

#include <sbml/packages/comp/sbml/Deletion.h>
#include <sbml/packages/comp/extension/CompExtension.h>

#include <sbml/SBMLDocument.h>

#include <check.h>



#include <sbml/common/extern.h>

LIBSBML_CPP_NAMESPACE_USE


BEGIN_C_DECLS

static Deletion_t *P;


void
DeletionTest_setup (void)
{
  P = Deletion_create(3, 1, 1);

  if (P == NULL)
  {
    fail("Deletion_create(3, 1, 1) returned a NULL pointer.");
  }
}


void
DeletionTest_teardown (void)
{
  Deletion_free(P);
}


START_TEST (test_comp_deletion_create)
{
  fail_unless( SBase_getTypeCode  ((SBase_t *) P) == SBML_COMP_DELETION );
  fail_unless( SBase_getMetaId    ((SBase_t *) P) == NULL );
  fail_unless( SBase_getNotes     ((SBase_t *) P) == NULL );
  fail_unless( SBase_getAnnotation((SBase_t *) P) == NULL );

  fail_unless( Deletion_getId     (P) == NULL );
  fail_unless( Deletion_getName   (P) == NULL );

  fail_unless( !Deletion_isSetId     (P) );
  fail_unless( !Deletion_isSetName   (P) );
}
END_TEST


START_TEST (test_comp_deletion_free_NULL)
{
  Deletion_free(NULL);
}
END_TEST


START_TEST (test_comp_deletion_id)
{
  const char *id = "mitochondria";


  fail_unless( !Deletion_isSetId(P) );
  
  Deletion_setId(P, id);

  fail_unless( !strcmp(Deletion_getId(P), id) );
  fail_unless( Deletion_isSetId(P) );

  if (Deletion_getId(P) == id)
  {
    fail("Deletion_setId(...) did not make a copy of string.");
  }
 
  Deletion_unsetId(P);
  
  fail_unless( !Deletion_isSetId(P) );

  if (Deletion_getId(P) != NULL)
  {
    fail("Deletion_unsetId(P) did not clear string.");
  }
}
END_TEST


START_TEST (test_comp_deletion_name)
{
  const char *name = "My_Favorite_Factory";


  fail_unless( !Deletion_isSetName(P) );

  Deletion_setName(P, name);

  fail_unless( !strcmp(Deletion_getName(P), name) );
  fail_unless( Deletion_isSetName(P) );

  if (Deletion_getName(P) == name)
  {
    fail("Deletion_setName(...) did not make a copy of string.");
  }

  Deletion_unsetName(P);
  
  fail_unless( !Deletion_isSetName(P) );

  if (Deletion_getName(P) != NULL)
  {
    fail("Deletion_unsetName(P) did not clear string.");
  }
}
END_TEST


START_TEST (test_comp_deletion_hasRequiredAttributes )
{
  Deletion_t *P = Deletion_create (3, 1, 1);

  fail_unless ( !Deletion_hasRequiredAttributes(P));

  Deletion_setId(P, "p");

  fail_unless ( !Deletion_hasRequiredAttributes(P));

  SBaseRef_setIdRef((SBaseRef_t*)(P), "s");
  
  fail_unless ( Deletion_hasRequiredAttributes(P));

  Deletion_free(P);
}
END_TEST


Suite *
create_suite_TestComp_Deletion (void)
{
  Suite *suite = suite_create("comp_deletion");
  TCase *tcase = tcase_create("comp_deletion");


  tcase_add_checked_fixture( tcase,
                             DeletionTest_setup,
                             DeletionTest_teardown );

  tcase_add_test( tcase, test_comp_deletion_create              );
  tcase_add_test( tcase, test_comp_deletion_free_NULL           );
  tcase_add_test( tcase, test_comp_deletion_id               );
  tcase_add_test( tcase, test_comp_deletion_name             );
  tcase_add_test( tcase, test_comp_deletion_hasRequiredAttributes        );

  suite_add_tcase(suite, tcase);

  return suite;
}

END_C_DECLS

