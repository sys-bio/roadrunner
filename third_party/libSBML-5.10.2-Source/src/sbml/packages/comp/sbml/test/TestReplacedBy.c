/**
 * \file    TestReplacedBy.c
 * \brief   ReplacedBy unit tests
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

#include <sbml/packages/comp/sbml/ReplacedBy.h>
#include <sbml/packages/comp/extension/CompExtension.h>

#include <sbml/SBMLDocument.h>

#include <check.h>



#include <sbml/common/extern.h>

LIBSBML_CPP_NAMESPACE_USE


BEGIN_C_DECLS

static ReplacedBy_t *P;


void
ReplacedByTest_setup (void)
{
  P = ReplacedBy_create(3, 1, 1);

  if (P == NULL)
  {
    fail("ReplacedBy_create(3, 1, 1) returned a NULL pointer.");
  }
}


void
ReplacedByTest_teardown (void)
{
  ReplacedBy_free(P);
}


START_TEST (test_comp_replacedBy_create)
{
  fail_unless( SBase_getTypeCode  ((SBase_t *) P) == SBML_COMP_REPLACEDBY );
  fail_unless( SBase_getMetaId    ((SBase_t *) P) == NULL );
  fail_unless( SBase_getNotes     ((SBase_t *) P) == NULL );
  fail_unless( SBase_getAnnotation((SBase_t *) P) == NULL );

  fail_unless( ReplacedBy_getSubmodelRef     (P) == NULL );

  fail_unless( !ReplacedBy_isSetSubmodelRef     (P) );
}
END_TEST


START_TEST (test_comp_replacedBy_free_NULL)
{
  ReplacedBy_free(NULL);
}
END_TEST


START_TEST (test_comp_replacedBy_submodelRef)
{
  const char *submodelRef = "mitochondria";


  fail_unless( !ReplacedBy_isSetSubmodelRef(P) );
  
  ReplacedBy_setSubmodelRef(P, submodelRef);

  fail_unless( !strcmp(ReplacedBy_getSubmodelRef(P), submodelRef) );
  fail_unless( ReplacedBy_isSetSubmodelRef(P) );

  if (ReplacedBy_getSubmodelRef(P) == submodelRef)
  {
    fail("ReplacedBy_setSubmodelRef(...) submodelRef not make a copy of string.");
  }
 
  ReplacedBy_unsetSubmodelRef(P);
  
  fail_unless( !ReplacedBy_isSetSubmodelRef(P) );

  if (ReplacedBy_getSubmodelRef(P) != NULL)
  {
    fail("ReplacedBy_unsetSubmodelRef(P) did not clear string.");
  }
}
END_TEST


START_TEST (test_comp_replacedBy_hasRequiredAttributes )
{
  ReplacedBy_t *P = ReplacedBy_create (3, 1, 1);

  fail_unless ( !ReplacedBy_hasRequiredAttributes(P));

  ReplacedBy_setSubmodelRef(P, "p");

  fail_unless ( !ReplacedBy_hasRequiredAttributes(P));

  SBaseRef_setIdRef((SBaseRef_t*)(P), "s");
  
  fail_unless ( ReplacedBy_hasRequiredAttributes(P));

  ReplacedBy_free(P);
}
END_TEST


Suite *
create_suite_TestComp_ReplacedBy (void)
{
  Suite *suite = suite_create("comp_replacedBy");
  TCase *tcase = tcase_create("comp_replacedBy");


  tcase_add_checked_fixture( tcase,
                             ReplacedByTest_setup,
                             ReplacedByTest_teardown );

  tcase_add_test( tcase, test_comp_replacedBy_create               );
  tcase_add_test( tcase, test_comp_replacedBy_free_NULL            );
  tcase_add_test( tcase, test_comp_replacedBy_submodelRef          );
  tcase_add_test( tcase, test_comp_replacedBy_hasRequiredAttributes);

  suite_add_tcase(suite, tcase);

  return suite;
}

END_C_DECLS

