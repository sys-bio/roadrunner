/**
 * \file    TestReplacedElement.c
 * \brief   ReplacedElement unit tests
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

#include <sbml/packages/comp/sbml/ReplacedElement.h>
#include <sbml/packages/comp/extension/CompExtension.h>

#include <sbml/SBMLDocument.h>

#include <check.h>



#include <sbml/common/extern.h>

LIBSBML_CPP_NAMESPACE_USE


BEGIN_C_DECLS

static ReplacedElement_t *P;


void
ReplacedElementTest_setup (void)
{
  P = ReplacedElement_create(3, 1, 1);

  if (P == NULL)
  {
    fail("ReplacedElement_create(3, 1, 1) returned a NULL pointer.");
  }
}


void
ReplacedElementTest_teardown (void)
{
  ReplacedElement_free(P);
}


START_TEST (test_comp_replacedElement_create)
{
  fail_unless( SBase_getTypeCode  ((SBase_t *) P) == SBML_COMP_REPLACEDELEMENT );
  fail_unless( SBase_getMetaId    ((SBase_t *) P) == NULL );
  fail_unless( SBase_getNotes     ((SBase_t *) P) == NULL );
  fail_unless( SBase_getAnnotation((SBase_t *) P) == NULL );

  fail_unless( ReplacedElement_getSubmodelRef (P) == NULL );
  fail_unless( ReplacedElement_getDeletion    (P) == NULL );
  fail_unless( ReplacedElement_getConversionFactor    (P) == NULL );

  fail_unless( !ReplacedElement_isSetSubmodelRef (P) );
  fail_unless( !ReplacedElement_isSetDeletion    (P) );
  fail_unless( !ReplacedElement_isSetConversionFactor    (P) );
}
END_TEST


START_TEST (test_comp_replacedElement_free_NULL)
{
  ReplacedElement_free(NULL);
}
END_TEST


START_TEST (test_comp_replacedElement_submodelRef)
{
  const char *submodelRef = "mitochondria";


  fail_unless( !ReplacedElement_isSetSubmodelRef(P) );
  
  ReplacedElement_setSubmodelRef(P, submodelRef);

  fail_unless( !strcmp(ReplacedElement_getSubmodelRef(P), submodelRef) );
  fail_unless( ReplacedElement_isSetSubmodelRef(P) );

  if (ReplacedElement_getSubmodelRef(P) == submodelRef)
  {
    fail("ReplacedElement_setSubmodelRef(...) did not make a copy of string.");
  }
 
  ReplacedElement_unsetSubmodelRef(P);
  
  fail_unless( !ReplacedElement_isSetSubmodelRef(P) );

  if (ReplacedElement_getSubmodelRef(P) != NULL)
  {
    fail("ReplacedElement_unsetSubmodelRef(P) did not clear string.");
  }
}
END_TEST


START_TEST (test_comp_replacedElement_deletion)
{
  const char *deletion = "My_Favorite_Factory";


  fail_unless( !ReplacedElement_isSetDeletion(P) );

  ReplacedElement_setDeletion(P, deletion);

  fail_unless( !strcmp(ReplacedElement_getDeletion(P), deletion) );
  fail_unless( ReplacedElement_isSetDeletion(P) );

  if (ReplacedElement_getDeletion(P) == deletion)
  {
    fail("ReplacedElement_setDeletion(...) did not make a copy of string.");
  }

  ReplacedElement_unsetDeletion(P);
  
  fail_unless( !ReplacedElement_isSetDeletion(P) );

  if (ReplacedElement_getDeletion(P) != NULL)
  {
    fail("ReplacedElement_unsetDeletion(P) did not clear string.");
  }
}
END_TEST


START_TEST (test_comp_replacedElement_conversionFactor)
{
  const char *conversionFactor = "My_Favorite_Factory";


  fail_unless( !ReplacedElement_isSetConversionFactor(P) );

  ReplacedElement_setConversionFactor(P, conversionFactor);

  fail_unless( !strcmp(ReplacedElement_getConversionFactor(P), conversionFactor) );
  fail_unless( ReplacedElement_isSetConversionFactor(P) );

  if (ReplacedElement_getConversionFactor(P) == conversionFactor)
  {
    fail("ReplacedElement_setConversionFactor(...) did not make a copy of string.");
  }

  ReplacedElement_unsetConversionFactor(P);
  
  fail_unless( !ReplacedElement_isSetConversionFactor(P) );

  if (ReplacedElement_getConversionFactor(P) != NULL)
  {
    fail("ReplacedElement_unsetConversionFactor(P) did not clear string.");
  }
}
END_TEST


START_TEST (test_comp_replacedElement_hasRequiredAttributes )
{
  ReplacedElement_t *P = ReplacedElement_create (3, 1, 1);

  fail_unless ( !ReplacedElement_hasRequiredAttributes(P));

  ReplacedElement_setSubmodelRef(P, "p");

  fail_unless ( !ReplacedElement_hasRequiredAttributes(P));

  SBaseRef_setIdRef((SBaseRef_t*)(P), "s");
  
  fail_unless ( ReplacedElement_hasRequiredAttributes(P));

  ReplacedElement_free(P);
}
END_TEST


Suite *
create_suite_TestComp_ReplacedElement (void)
{
  Suite *suite = suite_create("comp_replacedElement");
  TCase *tcase = tcase_create("comp_replacedElement");


  tcase_add_checked_fixture( tcase,
                             ReplacedElementTest_setup,
                             ReplacedElementTest_teardown );

  tcase_add_test( tcase, test_comp_replacedElement_create              );
  tcase_add_test( tcase, test_comp_replacedElement_free_NULL           );
  tcase_add_test( tcase, test_comp_replacedElement_submodelRef         );
  tcase_add_test( tcase, test_comp_replacedElement_deletion            );
  tcase_add_test( tcase, test_comp_replacedElement_conversionFactor    );
  tcase_add_test( tcase, test_comp_replacedElement_hasRequiredAttributes);

  suite_add_tcase(suite, tcase);

  return suite;
}

END_C_DECLS

