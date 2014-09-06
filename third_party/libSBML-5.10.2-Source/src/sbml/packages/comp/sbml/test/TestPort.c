/**
 * \file    TestPort.c
 * \brief   Port unit tests
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

#include <sbml/packages/comp/sbml/Port.h>
#include <sbml/packages/comp/extension/CompExtension.h>

#include <sbml/SBMLDocument.h>

#include <check.h>



#include <sbml/common/extern.h>

LIBSBML_CPP_NAMESPACE_USE


BEGIN_C_DECLS

static Port_t *P;


void
PortTest_setup (void)
{
  P = Port_create(3, 1, 1);

  if (P == NULL)
  {
    fail("Port_create(3, 1, 1) returned a NULL pointer.");
  }
}


void
PortTest_teardown (void)
{
  Port_free(P);
}


START_TEST (test_comp_port_create)
{
  fail_unless( SBase_getTypeCode  ((SBase_t *) P) == SBML_COMP_PORT );
  fail_unless( SBase_getMetaId    ((SBase_t *) P) == NULL );
  fail_unless( SBase_getNotes     ((SBase_t *) P) == NULL );
  fail_unless( SBase_getAnnotation((SBase_t *) P) == NULL );

  fail_unless( Port_getId     (P) == NULL );
  fail_unless( Port_getName   (P) == NULL );

  fail_unless( !Port_isSetId     (P) );
  fail_unless( !Port_isSetName   (P) );
}
END_TEST


START_TEST (test_comp_port_free_NULL)
{
  Port_free(NULL);
}
END_TEST


START_TEST (test_comp_port_id)
{
  const char *id = "mitochondria";


  fail_unless( !Port_isSetId(P) );
  
  Port_setId(P, id);

  fail_unless( !strcmp(Port_getId(P), id) );
  fail_unless( Port_isSetId(P) );

  if (Port_getId(P) == id)
  {
    fail("Port_setId(...) did not make a copy of string.");
  }
 
  Port_unsetId(P);
  
  fail_unless( !Port_isSetId(P) );

  if (Port_getId(P) != NULL)
  {
    fail("Port_unsetId(P) did not clear string.");
  }
}
END_TEST


START_TEST (test_comp_port_name)
{
  const char *name = "My_Favorite_Factory";


  fail_unless( !Port_isSetName(P) );

  Port_setName(P, name);

  fail_unless( !strcmp(Port_getName(P), name) );
  fail_unless( Port_isSetName(P) );

  if (Port_getName(P) == name)
  {
    fail("Port_setName(...) did not make a copy of string.");
  }

  Port_unsetName(P);
  
  fail_unless( !Port_isSetName(P) );

  if (Port_getName(P) != NULL)
  {
    fail("Port_unsetName(P) did not clear string.");
  }
}
END_TEST


START_TEST (test_comp_port_hasRequiredAttributes )
{
  Port_t *P = Port_create (3, 1, 1);

  fail_unless ( !Port_hasRequiredAttributes(P));

  Port_setId(P, "p");

  fail_unless ( !Port_hasRequiredAttributes(P));

  SBaseRef_setIdRef((SBaseRef_t*)(P), "s");
  
  fail_unless ( Port_hasRequiredAttributes(P));

  Port_free(P);
}
END_TEST


Suite *
create_suite_TestComp_Port (void)
{
  Suite *suite = suite_create("comp_port");
  TCase *tcase = tcase_create("comp_port");


  tcase_add_checked_fixture( tcase,
                             PortTest_setup,
                             PortTest_teardown );

  tcase_add_test( tcase, test_comp_port_create              );
  tcase_add_test( tcase, test_comp_port_free_NULL           );
  tcase_add_test( tcase, test_comp_port_id               );
  tcase_add_test( tcase, test_comp_port_name             );
  tcase_add_test( tcase, test_comp_port_hasRequiredAttributes        );

  suite_add_tcase(suite, tcase);

  return suite;
}

END_C_DECLS

