/**
 * @file    spec_example2.cpp
 * @brief   SBML hierarchical composition example
 * @author  Lucian Smith
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

#include <iostream>
#include <sstream>

#include <sbml/SBMLTypes.h>
#include <sbml/extension/SBMLExtensionRegister.h>
#include <sbml/extension/SBMLExtensionRegistry.h>

#include <sbml/packages/comp/extension/CompExtension.h>
#include <sbml/packages/comp/extension/CompSBasePlugin.h>
#include <sbml/packages/comp/extension/CompSBMLDocumentPlugin.h>
#include <sbml/packages/comp/extension/CompModelPlugin.h>

#ifdef WIN32
#include <conio.h>
#endif

LIBSBML_CPP_NAMESPACE_USE
using namespace std;

static SBMLExtensionRegister<CompExtension> compExtensionRegistry;

int main(int argc,char** argv)
{
  int retval = 0;
  SBMLNamespaces sbmlns(3,1,"comp",1);

  // create the document
  SBMLDocument *document = new SBMLDocument(&sbmlns);

  //Define the external model definitions
  CompSBMLDocumentPlugin* compdoc
      = static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);
  ExternalModelDefinition* extmod = compdoc->createExternalModelDefinition();
  extmod->setId("ExtMod1");
  extmod->setSource("enzyme_model.xml");
  extmod->setModelRef("enzyme");


  // create the main Model
  Model* model=document->createModel();
  
  // Set the submodels
  CompModelPlugin* mplugin = static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("A");
  submod1->setModelRef("ExtMod1");
  Submodel* submod2 = mplugin->createSubmodel();
  submod2->setId("B");
  submod2->setModelRef("ExtMod1");

  // create a replacement compartment
  Compartment* comp=model->createCompartment();
  comp->setSpatialDimensions((unsigned int)3);
  comp->setConstant(true);
  comp->setId("comp");
  comp->setSize(1L);

  //Tell the model that this compartment replaces both of the inside ones.
  CompSBasePlugin* compartplug = static_cast<CompSBasePlugin*>(comp->getPlugin("comp"));
  ReplacedElement re;
  re.setIdRef("comp");
  re.setSubmodelRef("A");
  compartplug->addReplacedElement(&re);
  re.setSubmodelRef("B");
  compartplug->addReplacedElement(&re);

  // create a replacement species
  Species* spec = model->createSpecies();
  spec->setCompartment("comp");
  spec->setHasOnlySubstanceUnits(false);
  spec->setConstant(false);
  spec->setBoundaryCondition(false);
  spec->setId("S");

  //Tell the model that this species replaces both of the inside ones.
  CompSBasePlugin* spp = static_cast<CompSBasePlugin*>(spec->getPlugin("comp"));
  re.setIdRef("S");
  re.setSubmodelRef("A");
  spp->addReplacedElement(&re);
  re.setSubmodelRef("B");
  spp->addReplacedElement(&re);

  writeSBMLToFile(document,"eg-import-external.xml");
  writeSBMLToFile(document,"spec_example2.xml");
  delete document;
  document = readSBMLFromFile("spec_example2.xml");
  if (document == NULL)
  {
    cout << "Error reading back in file." << endl;
    retval = -1;
  }
  else
  {
    document->setConsistencyChecks(LIBSBML_CAT_UNITS_CONSISTENCY, false);
    document->checkConsistency();
    if (document->getErrorLog()->getNumFailsWithSeverity(2) > 0
        || document->getErrorLog()->getNumFailsWithSeverity(3) > 0)
    {
      stringstream errorstream;
      document->printErrors(errorstream);
      cout << "Errors encoutered when round-tripping  SBML file: \n"
           <<  errorstream.str() << endl;
      retval = -1;
    }
    writeSBMLToFile(document, "spec_example2_rt.xml");
    delete document;
  }
#ifdef WIN32
  if (retval != 0)
  {
    cout << "(Press any key to exit.)" << endl;
    _getch();
  }
#endif
  return retval;
}
