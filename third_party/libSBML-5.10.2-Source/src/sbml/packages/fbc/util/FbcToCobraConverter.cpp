/**
 * @file    FbcToCobraConverter.cpp
 * @brief   Implementation of a fbc 2 cobra converter.
 * @author  Frank T. Bergmann
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
 * Copyright (C) 2009-2013 jointly by the following organizations: 
 *     1. California Institute of Technology, Pasadena, CA, USA
 *     2. EMBL European Bioinformatics Institute (EMBL-EBI), Hinxton, UK
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation.  A copy of the license agreement is provided
 * in the file named "LICENSE.txt" included with this software distribution
 * and also available online as http://sbml.org/software/libsbml/license.html
 * ---------------------------------------------------------------------- -->
 */


#include <sbml/packages/fbc/util/FbcToCobraConverter.h>
#include <sbml/conversion/SBMLConverterRegistry.h>
#include <sbml/conversion/SBMLConverterRegister.h>
#include <sbml/conversion/ConversionProperties.h>

#include <sbml/common/sbmlfwd.h>
#include <sbml/extension/SBasePlugin.h>
#include <sbml/extension/SBMLDocumentPlugin.h>
#include <sbml/SBMLWriter.h>
#include <sbml/SBMLReader.h>
#include <sbml/packages/fbc/common/FbcExtensionTypes.h>
#include <sbml/math/FormulaParser.h>

#ifdef __cplusplus

#include <algorithm>
#include <string>

using namespace std;
LIBSBML_CPP_NAMESPACE_BEGIN

/** @cond doxygenLibsbmlInternal */
/*
 * SBML Converter stuff below
 */
void FbcToCobraConverter::init()
{
  SBMLConverterRegistry::getInstance().addConverter(new FbcToCobraConverter());
}
/** @endcond */


FbcToCobraConverter::FbcToCobraConverter() 
 : SBMLConverter("SBML FBC to COBRA Converter")
{

}


FbcToCobraConverter::FbcToCobraConverter(const FbcToCobraConverter& orig) :
  SBMLConverter(orig)
{
}

FbcToCobraConverter* 
FbcToCobraConverter::clone() const
{
  return new FbcToCobraConverter(*this);
}


/*
 * Destroy this object.
 */
FbcToCobraConverter::~FbcToCobraConverter ()
{
}


ConversionProperties
  FbcToCobraConverter::getDefaultProperties() const
{
  static ConversionProperties prop;
  prop.addOption("convert fbc to cobra", true, "convert FBC L3V1 to SBML L2V4 with COBRA annotation");
  return prop;
}


bool 
  FbcToCobraConverter::matchesProperties(const ConversionProperties &props) const
{
  if (&props == NULL || !props.hasOption("convert fbc to cobra"))
    return false;
  return true;
}


void setObjectiveCoefficient(FbcModelPlugin* plugin, Model* model)
{
  if (plugin == NULL || model == NULL)
    return;

  Objective* obj = plugin->getActiveObjective();
  if (obj == NULL)
    return;

  for (unsigned int i = 0; i < obj->getNumFluxObjectives(); ++i)
  {
    FluxObjective* fluxObj = obj->getFluxObjective(i);
    if (fluxObj == NULL)
      continue;
    Reaction* reaction = model->getReaction(fluxObj->getReaction());
    if (reaction == NULL)
      continue;
    KineticLaw* law = reaction->getKineticLaw();
    if (law == NULL)
      continue;
    LocalParameter* param = law->getLocalParameter("OBJECTIVE_COEFFICIENT");
    param->setValue(fluxObj->getCoefficient());
  }
}


void createKineticLawForReaction(Reaction* reaction)
{
  if (reaction == NULL)
    return;
  reaction->unsetKineticLaw();
  KineticLaw *law = reaction->getKineticLaw();
  if (law == NULL)
  {
    law = reaction->createKineticLaw();
    LocalParameter* fluxValue = law->createLocalParameter();
    fluxValue->initDefaults();
    fluxValue->setId("FLUX_VALUE");
    fluxValue->setValue(0);
    fluxValue->setUnits("dimensionless");
    law->setMath(SBML_parseFormula("FLUX_VALUE"));
  }

  LocalParameter* LB = law->getLocalParameter("LOWER_BOUND");
  if (LB == NULL)
  {
    LB = law->createLocalParameter();
    LB->initDefaults();
    LB->setId("LOWER_BOUND");
    LB->setUnits("dimensionless");
    LB->setValue(-std::numeric_limits<double>::infinity());
  }

  LocalParameter* UB = law->getLocalParameter("UPPER_BOUND");
  if (UB == NULL)
  {
    UB = law->createLocalParameter();
    UB->initDefaults();
    UB->setId("UPPER_BOUND");
    UB->setUnits("dimensionless");
    LB->setValue(std::numeric_limits<double>::infinity());
  }

  LocalParameter* param = law->getLocalParameter("OBJECTIVE_COEFFICIENT");
  if (param == NULL)
  {
    param = law->createLocalParameter();
    param->initDefaults();
    param->setId("OBJECTIVE_COEFFICIENT");
    param->setUnits("dimensionless");
    param->setValue(0);
  }

}

void updateKineticLawFromBound(Reaction* reaction, FluxBound* current)
{
  if (reaction == NULL || current == NULL)
    return;
  const string operation = current -> getOperation();

  KineticLaw *law = reaction->getKineticLaw();
  LocalParameter* LB = law->getLocalParameter("LOWER_BOUND");
  LocalParameter* UB = law->getLocalParameter("UPPER_BOUND");

  if (operation == "less" || operation == "lessEqual" || operation == "equal")
  {
    UB->setValue(current->getValue());
  }
  if (operation == "greater" || operation == "greaterEqual" || operation == "equal")
  {
    LB->setValue(current->getValue());
  }
  
}

string getNotesForFormula(const string& formula)
{
  stringstream str; 

  str 
    << "<html xmlns=\"http://www.w3.org/1999/xhtml\">\n\t<p>FORMULA: "
    << formula
    << "</p>\n</html>";

  return str.str();
}

int 
  FbcToCobraConverter::convert()
{  
  int result = LIBSBML_OPERATION_FAILED;

  if (mDocument == NULL) 
  {
    return LIBSBML_INVALID_OBJECT;
  }

  Model* mModel = mDocument->getModel();
  if (mModel == NULL) 
  {
    return LIBSBML_INVALID_OBJECT;
  }

  FbcModelPlugin *plugin =
    static_cast<FbcModelPlugin*>(mDocument->getModel()->getPlugin("fbc"));

  // if we have don't have a fbc model we cannot do the conversion
  if (plugin == NULL || mDocument->getLevel() != 3)
  {
    return LIBSBML_OPERATION_FAILED;
  }

  // collect information

  Model* model = mDocument->getModel();
  map<const string, int> chargeMap;
  map<const string, string> formulaMap;

  for (unsigned int i = 0; i < model->getNumSpecies(); ++i)
  {
    Species* current = model->getSpecies(i);
    const string& currentId = current->getId();
    FbcSpeciesPlugin *splugin = static_cast<FbcSpeciesPlugin*>(current->getPlugin("fbc"));
    if (splugin == NULL)
      continue;
    if (splugin->isSetCharge())
    {
      chargeMap[currentId] = splugin->getCharge();
    }
    if (splugin->isSetChemicalFormula())
    {
      formulaMap[currentId] = splugin->getChemicalFormula();
    }
  }

  // create KineticLaw
  for (unsigned int i = 0; i < model->getNumReactions(); ++i)
  {
    Reaction* reaction = model->getReaction(i);
    if (reaction == NULL)
      continue;

    createKineticLawForReaction(reaction);

  }

  // update kinetic law from bounds
  for (unsigned int i = 0; i < plugin->getNumFluxBounds(); ++i)
  {
    FluxBound *current = plugin->getFluxBound(i);
    if (current == NULL)
      continue;
    Reaction* reaction = model->getReaction(current->getReaction());
    if (reaction == NULL)
      continue;

    updateKineticLawFromBound(reaction, current);

  }

  setObjectiveCoefficient(plugin, model);

  // disable package
  mDocument->enablePackage("http://www.sbml.org/sbml/level3/version1/fbc/version1", "fbc",false);

  // convert model to L2V1 (as L2V2 is the last model that had charge)
  mDocument->setConversionValidators(AllChecksON & UnitsCheckOFF);
  
  ConversionProperties prop(new SBMLNamespaces(2,1));
  prop.addOption("strict", false, "should validity be preserved");
  prop.addOption("ignorePackages", true, "convert even if packages are used");
  prop.addOption("setLevelAndVersion", true, "convert the document to the given level and version");
  int conversionResult = mDocument->convert(prop);
  if (conversionResult != LIBSBML_OPERATION_SUCCESS)
    return conversionResult;

  // set charge on species
  for (unsigned int i = 0; i < model->getNumSpecies(); ++i)  
  {
    Species* current = model->getSpecies(i);
    const string currentId = current->getId();
    int charge = chargeMap[currentId];

    if (charge != 0)
      current->setCharge(charge);

    const string formula = formulaMap[currentId];
    if (!formula.empty())
    {
      current->setNotes( getNotesForFormula(formula) );
    }
  }


  result = LIBSBML_OPERATION_SUCCESS;
  return result;
}


/** @cond doxygenIgnored */


/** @endcond */

LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */


