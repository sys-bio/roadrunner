/**
* @file    flattenModel.c
* @brief   Flattens the comp code from the given SBML file.
* @author  Sarah Keating
 *
 * <!--------------------------------------------------------------------------
 * This sample program is distributed under a different license than the rest
 * of libSBML.  This program uses the open-source MIT license, as follows:
 *
 * Copyright (c) 2013-2014 by the California Institute of Technology
 * (California, USA), the European Bioinformatics Institute (EMBL-EBI, UK)
 * and the University of Heidelberg (Germany), with support from the National
 * Institutes of Health (USA) under grant R01GM070923.  All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Neither the name of the California Institute of Technology (Caltech), nor
 * of the European Bioinformatics Institute (EMBL-EBI), nor of the University
 * of Heidelberg, nor the names of any contributors, may be used to endorse
 * or promote products derived from this software without specific prior
 * written permission.
 * ------------------------------------------------------------------------ -->
*/


#include <stdio.h>
#include <sbml/SBMLTypes.h>
#include <sbml/conversion/ConversionProperties.h>


int
  main (int argc, char *argv[])
{
  SBMLDocument_t *doc;
  int leavePorts = 0;

  if (argc < 3 || argc > 4)
  {
    printf("Usage: flattenModel [-p] input-filename output-filename\n-p:list unused ports\n");
    return 2;
  }

  if (argc == 3)
  {
    doc = readSBML(argv[1]);
  }
  else
  {
    if (strcmp(argv[1], "-p") != 0)
    {
      printf("Usage: flattenModel [-p] input-filename output-filename\n-p:list unused ports\n");
      return 2;
    }
    doc = readSBML(argv[2]);
    leavePorts = 1;
  }


  /* if there are errors do not try to flatten */
  if (SBMLDocument_getNumErrorsWithSeverity(doc, LIBSBML_SEV_ERROR) > 0)
  {
    SBMLDocument_printErrors(doc, stderr);
  }
  else
  {
    /* need new variables ... */
    ConversionProperties_t* props;
    ConversionOption_t* option1;
    ConversionOption_t* option2;

    /* create a new conversion properties structure */
    props = ConversionProperties_create();

    /* add an option that we want to flatten comp */
    option1 = ConversionOption_create("flatten comp");
    ConversionOption_setType(option1, CNV_TYPE_BOOL);
    ConversionOption_setValue(option1, "true");
    ConversionOption_setDescription(option1, "flatten comp");
    ConversionProperties_addOption(props, option1);

    /* add an option to leave ports if the user has requested this */
    if (leavePorts == 1)
    {
      option2 = ConversionOption_create("leavePorts");
      ConversionOption_setType(option2, CNV_TYPE_BOOL);
      ConversionOption_setValue(option2, "true");
      ConversionOption_setDescription(option2, "unused ports should be listed in the flattened model");
      ConversionProperties_addOption(props, option2);
    }

    /* perform the conversion */
    if (SBMLDocument_convert(doc, props) != LIBSBML_OPERATION_SUCCESS)
    {
      printf ("conversion failed ... ");
	  SBMLDocument_printErrors(doc, stderr);
      return 3;
    }

    /* successfully completed, write the resulting file */
    if (argc == 4)
    {
      writeSBML(doc, argv[3]);
    }
    else
    {
      writeSBML(doc, argv[2]);
    }
  }

  return 0;
}

