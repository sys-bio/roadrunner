/**
 * \file    TestRunner.c
 * \brief   Runs all unit tests in the comp util module
 * \author  Frank T. Bergmann
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

#include <check.h>
#include <stdlib.h>
#include <string.h>

#include <sbml/util/memory.h>
#include <sbml/common/extern.h>


/* START: workaround for VS2013 bug */
/* from https://connect.microsoft.com/VisualStudio/feedback/details/806362/ */

#if defined(_M_IX86) && _MSC_VER >= 1800 && _MSC_VER < 1900

#pragma warning(disable:4075) // C4075: warning about unknown initializer section
#pragma init_seg( ".CRT$XCB" ) // section sorts alphabetically before any user code section

extern "C" void _except1();
extern "C" void _ftol3_except();
extern "C" int __stdcall VirtualProtect( void*, int, int, int* );

// ABI specifies empty FPU stack on entry and exit of a non-inlined function
static __declspec(noinline) unsigned __int32 ConvertInfinityToUint32()
{
    volatile double zero = 0.0;
    return (unsigned __int32)(1.0 / zero);
}

static __declspec(naked) void HackedEndOfFtol3Except()
{
    _asm call _except1
    _asm fstp st( 0 )
    _asm add esp, 0x20
    _asm ret
}

static bool PatchVS2013FPUStackBug()
{
    // If the top-of-stack bits of the status word changed, then converting
    // an infinity or NaN to uint32 has leaked an FPU stack entry.
    volatile short swBefore, swAfter;
    _asm fnstsw swBefore
    ConvertInfinityToUint32();
    _asm fnstsw swAfter
    if ( (swBefore & 0x3800) == (swAfter & 0x3800) )
        return true; // already patched, either by MS or by this hack

    // Fix the FPU stack, then fix the _ftol3_except function itself
    _asm fstp st( 0 )

    char* p = (char*)&_ftol3_except;
    // Edit-and-continue builds use a relative jump to protect the actual function.
    if ( p[0] == '\xE9' )
    {
        p = p + 5 + *(int*)(p + 1);
    }
    // The _ftol3_except function begins with SUB ESP, 20 then WAIT; call to _except1 is at 57 bytes
    // (relative offset for call is bytes 58-62) followed by ADD ESP,0x20 then the final RET.
    if ( p[0] == '\x83' && p[1] == '\xEC' && p[2] == '\x20' && p[3] == '\x9B' &&
        p[57] == '\xE8' && p[62] == '\x83' && p[63] == '\xC4' && p[64] == '\x20' && p[65] == '\xC3' )
    {
        // Replace call to _except1 with a jump to HackedEndOfFtol3Except
        int old = 0x20;
        VirtualProtect( p + 57, 5, 0x40, &old );
        p[57] = '\xE9';
        *(int*)(p + 58) = (int)&HackedEndOfFtol3Except - (int)(p + 62);
        VirtualProtect( p + 57, 5, old, &old );
        return true;
    }
    return false;
}

#pragma comment(linker, "/include:_g_bPatchedVS2013FPUStackBug")
extern "C" bool g_bPatchedVS2013FPUStackBug = PatchVS2013FPUStackBug();

#endif

/* END: workaround for VS2013 bug */ 



/**
 * Test suite creation function prototypes.
 *
 * These functions are needed only for calls in main() below.  Therefore a
 * separate header file is not necessary and only adds a maintenance burden
 * to keep the two files synchronized.
 */
#if defined(__cplusplus)
LIBSBML_CPP_NAMESPACE_USE
CK_CPPSTART
#endif

Suite *create_suite_TestFlatteningConverter  (void);
Suite *create_suite_TestFlatteningUnknownPackageRefs  (void);
Suite *create_suite_TestFlatteningErrorMessages  (void);
Suite *create_suite_TestURIResolvers (void);
Suite *create_suite_TestExternalModelResolving (void);
Suite *create_suite_TestCompFlatteningNewFlags (void);

/**
 * Global.
 *
 * Declared extern in TestReadFromFileN suites.
 */
char *TestDataDirectory;


/**
 * Sets TestDataDirectory for the the TestReadFromFileN suites.
 *
 * For Automake's distcheck target to work properly, TestDataDirectory must
 * begin with the value of the environment variable SRCDIR.
 */
void
setTestDataDirectory (void)
{
  char *srcdir = getenv("srcdir");
  int  length  = (srcdir == NULL) ? 0 : strlen(srcdir);


  /**
   * strlen("/test-data/") = 11 + 1 (for NULL) = 12
   */
  TestDataDirectory = (char *) safe_calloc( length + 12, sizeof(char) );

  if (srcdir != NULL)
  {
    strcpy(TestDataDirectory, srcdir);
    strcat(TestDataDirectory, "/");
  }

  strcat(TestDataDirectory, "test-data/");
}


int
main (void) 
{ 
  int num_failed;

  setTestDataDirectory();

  SRunner *runner = srunner_create( create_suite_TestURIResolvers() );
  srunner_add_suite( runner, create_suite_TestFlatteningUnknownPackageRefs() );
  srunner_add_suite( runner, create_suite_TestFlatteningErrorMessages() );
  srunner_add_suite( runner, create_suite_TestFlatteningConverter() );
  srunner_add_suite( runner, create_suite_TestExternalModelResolving() );
  srunner_add_suite( runner, create_suite_TestCompFlatteningNewFlags() );

  /* srunner_set_fork_status(runner, CK_NOFORK); */

  srunner_run_all(runner, CK_NORMAL);
  num_failed = srunner_ntests_failed(runner);

  srunner_free(runner);

  return num_failed;
}

#if defined(__cplusplus)
CK_CPPEND
#endif


