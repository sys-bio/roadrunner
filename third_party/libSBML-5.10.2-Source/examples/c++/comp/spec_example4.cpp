/**
 * @file    spec_example4.cpp
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
  CompSBMLDocumentPlugin* compdoc
      = static_cast<CompSBMLDocumentPlugin*>(document->getPlugin("comp"));
  compdoc->setRequired(true);

  // create the 'enzyme' model definition
  ModelDefinition* mod1 = compdoc->createModelDefinition();
  mod1->setId("enzyme");
  mod1->setName("enzyme");
  Compartment* comp=mod1->createCompartment();
  comp->setSpatialDimensions((unsigned int)3);
  comp->setConstant(true);
  comp->setId("comp");
  comp->setSize(1L);

  // We have to construct it this way because we get the
  // comp plugin from it later.
  Species spec(&sbmlns);
  spec.setCompartment("comp");
  spec.setHasOnlySubstanceUnits(false);
  spec.setConstant(false);
  spec.setBoundaryCondition(false);
  spec.setId("S");
  mod1->addSpecies(&spec);
  spec.setId("E");
  mod1->addSpecies(&spec);
  spec.setId("D");
  mod1->addSpecies(&spec);
  spec.setId("ES");
  mod1->addSpecies(&spec);

  Reaction rxn(&sbmlns);
  rxn.setReversible(true);
  rxn.setFast(false);
  Reaction rxn2(rxn);
  rxn.setId("J0");
  rxn2.setId("J1");
  SpeciesReference sr(&sbmlns);
  sr.setConstant(true);
  sr.setStoichiometry(1);
  sr.setSpecies("S");
  rxn.addReactant(&sr);
  sr.setSpecies("E");
  rxn.addReactant(&sr);
  rxn2.addProduct(&sr);
  sr.setSpecies("ES");
  rxn.addProduct(&sr);
  rxn2.addReactant(&sr);
  sr.setSpecies("D");
  rxn2.addProduct(&sr);

  mod1->addReaction(&rxn);
  mod1->addReaction(&rxn2);

  CompModelPlugin* mod1plug
      = static_cast<CompModelPlugin*>(mod1->getPlugin("comp"));
  Port m1port;
  m1port.setIdRef("comp");
  m1port.setId("comp_port");
  mod1plug->addPort(&m1port);
  m1port.setIdRef("S");
  m1port.setId("S_port");
  mod1plug->addPort(&m1port);
  m1port.setIdRef("E");
  m1port.setId("E_port");
  mod1plug->addPort(&m1port);
  m1port.setIdRef("D");
  m1port.setId("D_port");
  mod1plug->addPort(&m1port);
  m1port.setIdRef("ES");
  m1port.setId("ES_port");
  mod1plug->addPort(&m1port);
  m1port.setIdRef("J0");
  m1port.setId("J0_port");
  mod1plug->addPort(&m1port);
  m1port.setIdRef("J1");
  m1port.setId("J1_port");
  mod1plug->addPort(&m1port);

  //Create the 'simple' model definition
  ModelDefinition* mod2 = compdoc->createModelDefinition();
  mod2->setId("simple");
  Compartment* comp2=mod2->createCompartment();
  comp2->setSpatialDimensions((unsigned int)3);
  comp2->setConstant(true);
  comp2->setId("comp");
  comp2->setSize(1L);

  spec.setCompartment("comp");
  spec.setHasOnlySubstanceUnits(false);
  spec.setConstant(false);
  spec.setBoundaryCondition(false);
  spec.setId("S");
  spec.setInitialConcentration(5);
  mod2->addSpecies(&spec);
  spec.setId("D");
  spec.setInitialConcentration(10);
  mod2->addSpecies(&spec);

  // We have to construct it this way because we get the
  // comp plugin from it later.
  Reaction rxn3(&sbmlns);
  rxn3.setReversible(true);
  rxn3.setFast(false);
  rxn3.setId("J0");

  SpeciesReference sr2(&sbmlns); //These will not need plugins.
  sr2.setConstant(true);
  sr2.setStoichiometry(1);
  sr2.setSpecies("S");
  rxn3.addReactant(&sr2);
  sr2.setSpecies("D");
  rxn3.addProduct(&sr2);

  mod2->addReaction(&rxn3);

  CompModelPlugin* mod2plug
      = static_cast<CompModelPlugin*>(mod2->getPlugin("comp"));
  Port port;
  port.setId("S_port");
  port.setIdRef("S");
  mod2plug->addPort(&port);

  Port* port2 = mod2plug->createPort();
  port2->setId("D_port");
  port2->setIdRef("D");

  port.setId("comp_port");
  port.setIdRef("comp");
  mod2plug->addPort(&port);

  port.setId("J0_port");
  port.setIdRef("J0");
  mod2plug->addPort(&port);

  // create the main Model
  Model* model=document->createModel();
  model->setId("complexified");
  
  // Set the submodels
  CompModelPlugin* mplugin
      = static_cast<CompModelPlugin*>(model->getPlugin("comp"));
  Submodel* submod1 = mplugin->createSubmodel();
  submod1->setId("A");
  submod1->setModelRef("enzyme");
  Submodel* submod2 = mplugin->createSubmodel();
  submod2->setId("B");
  submod2->setModelRef("simple");
  Deletion* del = submod2->createDeletion();
  del->setPortRef("J0_port");
  del->setId("oldrxn");

  // Synchronize the compartments
  Compartment* mcomp=model->createCompartment();
  mcomp->setSpatialDimensions((unsigned int)3);
  mcomp->setConstant(true);
  mcomp->setId("comp");
  mcomp->setSize(1L);
  CompSBasePlugin* compartplug
      = static_cast<CompSBasePlugin*>(mcomp->getPlugin("comp"));
  ReplacedElement re;
  re.setIdRef("comp");
  re.setSubmodelRef("A");
  compartplug->addReplacedElement(&re);
  re.setSubmodelRef("B");
  re.unsetIdRef();
  re.setPortRef("comp_port");
  compartplug->addReplacedElement(&re);

  //Synchronize the species
  spec.setId("S");
  spec.unsetInitialConcentration();
  CompSBasePlugin* specplug
      = static_cast<CompSBasePlugin*>(spec.getPlugin("comp"));
  ReplacedElement* sre = specplug->createReplacedElement();
  sre->setSubmodelRef("A");
  sre->setIdRef("S");
  ReplacedBy* srb = specplug->createReplacedBy();
  srb->setSubmodelRef("B");
  srb->setPortRef("S_port");
  model->addSpecies(&spec);

  spec.setId("D");
  sre->setIdRef("D");
  srb->setPortRef("D_port");
  model->addSpecies(&spec);

  //Synchronize the reactions
  Reaction blankrxn(&sbmlns);
  blankrxn.setId("J0");
  blankrxn.setReversible(true);
  blankrxn.setFast(false);
  SpeciesReference sr3(&sbmlns); //These will not need plugins.
  sr3.setConstant(true);
  sr3.setStoichiometry(1);
  sr3.setSpecies("S");
  blankrxn.addReactant(&sr3);
  CompSBasePlugin* blankrxnplug
      = static_cast<CompSBasePlugin*>(blankrxn.getPlugin("comp"));
  ReplacedElement deletion;
  deletion.setDeletion("oldrxn");
  deletion.setSubmodelRef("B");
  blankrxnplug->addReplacedElement(&deletion);
  ReplacedBy* rb2 = blankrxnplug->createReplacedBy();
  rb2->setSubmodelRef("A");
  rb2->setPortRef("J0_port");
  model->addReaction(&blankrxn);
  blankrxn.removeReactant(0);
  sr3.setSpecies("D");
  blankrxn.addProduct(&sr3);
  blankrxn.setId("J1");
  rb2->setPortRef("J1_port");
  model->addReaction(&blankrxn);

  writeSBMLToFile(document,"eg-replacement.xml");
  writeSBMLToFile(document,"spec_example4.xml");
  delete document;
  document = readSBMLFromFile("spec_example4.xml");
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
    writeSBMLToFile(document, "spec_example4_rt.xml");
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
