/**
 * @file    TestReadFbcExtension.cpp
 * @brief   Unit tests of writing FbcExtension 
 * @author  Fank T. Bergmann
 *
 * $Id: $
 * $HeadURL: $
 */

#include <limits>

#include <iostream>
#include <check.h>
#include <sbml/SBMLTypes.h>
#include <sbml/extension/SBMLExtensionRegistry.h>
#include <sbml/conversion/ConversionProperties.h>
#include <sbml/packages/fbc/extension/FbcModelPlugin.h>
#include <sbml/packages/fbc/extension/FbcExtension.h>
#include <sbml/packages/fbc/common/FbcExtensionTypes.h>
#include <sbml/packages/fbc/validator/FbcSBMLError.h>
#include <string>

/** @cond doxygenIgnored */

using namespace std;
LIBSBML_CPP_NAMESPACE_USE

/** @endcond doxygenIgnored */


CK_CPPSTART

extern char *TestDataDirectory;

START_TEST (test_FbcExtension_read_L3V1V1)
{
  char *filename = safe_strcat(TestDataDirectory, "fbc_example1.xml");
  SBMLDocument *document = readSBMLFromFile(filename);
  
  fail_unless(document->getPackageName() == "core");
  
  Model *model = document->getModel();
  
  fail_unless(model != NULL);
  fail_unless(model->getPackageName() == "core");
  fail_unless(document->getNumErrors() == 0);
  
  // get the fbc plugin
  
  FbcModelPlugin* mplugin = static_cast<FbcModelPlugin*>(model->getPlugin("fbc"));
  fail_unless(mplugin != NULL);
  
  fail_unless(mplugin->getNumObjectives() == 1);
  fail_unless(mplugin->getListOfObjectives()->getPackageName() == "fbc");
  
  Objective* objective = mplugin->getObjective(0);
  fail_unless(objective->getId() == "obj1");
  fail_unless(objective->getType() == "maximize");
  fail_unless(objective->getNumFluxObjectives()  == 1);
  fail_unless(objective->getPackageName() == "fbc");
  
  fail_unless(objective->getListOfFluxObjectives()->getPackageName() == "fbc");
  
  FluxObjective* fluxObjective = objective->getFluxObjective(0);
  fail_unless(fluxObjective->getReaction()      == "J8");
  fail_unless(fluxObjective->getPackageName() == "fbc");
  fail_unless(fluxObjective->getCoefficient() == 1);
  
  fail_unless(mplugin->getNumFluxBounds() == 1);
  fail_unless(mplugin->getListOfFluxBounds()->getPackageName() == "fbc");
  
  FluxBound* bound = mplugin->getFluxBound(0);
  fail_unless(bound != NULL);
  
  fail_unless(bound->getId()      == "bound1");
  fail_unless(bound->getPackageName() == "fbc");
  fail_unless(bound->getReaction() == "J0");
  fail_unless(bound->getOperation() == "equal");
  fail_unless(bound->getValue() == 10);
  
  
  delete document;  
}
END_TEST


START_TEST (test_FbcExtension_read_L3V1V1_defaultNS)
{
  char *filename = safe_strcat(TestDataDirectory, "fbc_example1_defaultNS.xml");
  
  SBMLDocument *document = readSBMLFromFile(filename);
  
  fail_unless(document->getPackageName() == "core");
  
  Model *model = document->getModel();
  
  fail_unless(model != NULL);
  fail_unless(model->getPackageName() == "core");
  fail_unless(document->getNumErrors() == 0);
  
  // get the fbc plugin
  
  FbcModelPlugin* mplugin = static_cast<FbcModelPlugin*>(model->getPlugin("fbc"));
  fail_unless(mplugin != NULL);
  
  fail_unless(mplugin->getNumObjectives() == 1);
  fail_unless(mplugin->getListOfObjectives()->getPackageName() == "fbc");
  
  Objective* objective = mplugin->getObjective(0);
  fail_unless(objective->getId() == "obj1");
  fail_unless(objective->getType() == "maximize");
  fail_unless(objective->getNumFluxObjectives()  == 1);
  fail_unless(objective->getPackageName() == "fbc");
  
  fail_unless(objective->getListOfFluxObjectives()->getPackageName() == "fbc");
  
  FluxObjective* fluxObjective = objective->getFluxObjective(0);
  fail_unless(fluxObjective->getReaction()      == "J8");
  fail_unless(fluxObjective->getPackageName() == "fbc");
  fail_unless(fluxObjective->getCoefficient() == 1);
  
  fail_unless(mplugin->getNumFluxBounds() == 1);
  fail_unless(mplugin->getListOfFluxBounds()->getPackageName() == "fbc");
  
  FluxBound* bound = mplugin->getFluxBound(0);
  fail_unless(bound->getId()      == "bound1");
  fail_unless(bound->getPackageName() == "fbc");
  fail_unless(bound->getReaction() == "J0");
  fail_unless(bound->getOperation() == "equal");
  fail_unless(bound->getValue() == 10);
  
  
  delete document;  
}
END_TEST


START_TEST (test_FbcExtension_read_L3V1V1_unknown_elements)
{
  const char* s1 =
  "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
  "<sbml xmlns=\"http://www.sbml.org/sbml/level3/version1/core\" xmlns:fbc=\"http://www.sbml.org/sbml/level3/version1/fbc/version1\" level=\"3\" version=\"1\" fbc:required=\"false\">\n"
  "  <model>\n"
  "    <listOfCompartments>\n"
  "      <compartment id=\"cytosol\" constant=\"true\"/>\n"
  "    </listOfCompartments>\n"
  "    <listOfSpecies>\n"
  "      <species id=\"ATPm\" compartment=\"mitochon\" initialConcentration=\"2\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n"
  "    </listOfSpecies>\n"
  "    <listOfFluxBounds xmlns=\"http://www.sbml.org/sbml/level3/version1/fbc/version1\">\n"
  "       <fluxBound id=\"bound1\" reaction=\"J0\" operation=\"equal\" value=\"10\" unkown=\"something\"/>\n"
  "     </listOfFluxBounds>\n"
  "     <listOfObjectives xmlns=\"http://www.sbml.org/sbml/level3/version1/fbc/version1\" activeObjective=\"obj1\">\n"
  "       <objective id=\"obj1\" type=\"maximize\" unkown=\"something\">\n"
  "         <listOfFluxes>\n"
  "           <fluxObjective reaction=\"J8\" coefficient=\"1\" unkown=\"something\"/>\n"
  "         </listOfFluxes>\n"
  "       </objective>\n"
  "     </listOfObjectives>\n"
  "  </model>\n"
  "</sbml>\n"
  ;
  
  SBMLDocument *document = readSBMLFromString(s1);
  Model *model = document->getModel();
  
  //document->printErrors();
  //cout << document->getNumErrors() << endl;
  
  fail_unless(model != NULL);
  fail_unless(document->getNumErrors() == 3);
  
  delete document;
}
END_TEST

START_TEST (test_FbcExtension_OpertationTypes)
{
  // it seems we will drop the strict bounds, without breaking the models
  // so we collapse the types
  fail_unless(FluxBoundOperation_fromString("less") == FluxBoundOperation_fromString("lessEqual"));
  fail_unless(FluxBoundOperation_fromString("greater") == FluxBoundOperation_fromString("greaterEqual"));
  
  fail_unless(strcmp(FluxBoundOperation_toString(FLUXBOUND_OPERATION_GREATER),
                     FluxBoundOperation_toString(FLUXBOUND_OPERATION_GREATER_EQUAL)) == 0);
  fail_unless(strcmp(FluxBoundOperation_toString(FLUXBOUND_OPERATION_LESS),
                     FluxBoundOperation_toString(FLUXBOUND_OPERATION_LESS_EQUAL)) == 0);

}
END_TEST

START_TEST(test_FbcExtension_read_L3V1V1_with_wonky_chemicals)
{
  const char* modelString = "<?xml version='1.0' encoding='UTF-8'?>"
"<sbml xmlns:html='http://www.w3.org/1999/xhtml' xmlns='http://www.sbml.org/sbml/level3/version1/core' xmlns:fbc='http://www.sbml.org/sbml/level3/version1/fbc/version1' level='3' version='1' fbc:required='false'>"
"  <model metaid='xxx' id='yyy' timeUnits='dimensionless'>"
"    <listOfCompartments>"
"      <compartment id='Internal_Species' name='Internal_Species' spatialDimensions='3' size='1' constant='false'/>"
"      <compartment id='External_Species' name='External_Species' spatialDimensions='3' size='1' constant='false'/>"
"    </listOfCompartments>"
"    <listOfSpecies>"
"      <species metaid='meta_B1' id='B1' name='Protein component' compartment='Internal_Species' initialConcentration='0' hasOnlySubstanceUnits='false' boundaryCondition='false' constant='false' fbc:chemicalFormula='PROTEIN COMPONENT'>"
"        <notes>"
"          <html:p>chemFormula: Protein component</html:p>"
"        </notes>"
"        <annotation>"
"          <listOfKeyValueData xmlns='http://pysces.sourceforge.net/KeyValueData'>"
"            <data id='chemFormula' type='string' value='Protein component'/>"
"          </listOfKeyValueData>"
"        </annotation>"
"      </species>"
"      <species metaid='meta_B2' id='B2' name='DNA component' compartment='Internal_Species' initialConcentration='0' hasOnlySubstanceUnits='false' boundaryCondition='false' constant='false' fbc:chemicalFormula='DNA COMPONENT'>"
"        <notes>"
"          <html:p>chemFormula: DNA component</html:p>"
"        </notes>"
"        <annotation>"
"          <listOfKeyValueData xmlns='http://pysces.sourceforge.net/KeyValueData'>"
"            <data id='chemFormula' type='string' value='DNA component'/>"
"          </listOfKeyValueData>"
"        </annotation>"
"      </species>"
"    </listOfSpecies>"
"    <listOfReactions>"
"      <reaction metaid='meta_R00192' id='R00192' name='R106' reversible='false' fast='false'>"
"        <notes>"
"          <html:p>EC Number: 3.3.1.1</html:p>"
"          <html:p>GENE ASSOCIATION: sll1234</html:p>"
"        </notes>"
"        <annotation>"
"          <listOfKeyValueData xmlns='http://pysces.sourceforge.net/KeyValueData'>"
"            <data id='EC_Number' type='string' value='3.3.1.1'/>"
"            <data id='GENE_ASSOCIATION' type='string' value='sll1234'/>"
"          </listOfKeyValueData>"
"        </annotation>"
"        <listOfReactants>"
"          <speciesReference species='B1' stoichiometry='1' constant='true'/>"
"        </listOfReactants>"
"        <listOfProducts>"
"          <speciesReference species='B2' stoichiometry='1' constant='true'/>"
"        </listOfProducts>"
"      </reaction>"
"    </listOfReactions>"
"    <fbc:listOfFluxBounds>"
"      <fbc:fluxBound fbc:reaction='R00192' fbc:operation='greaterEqual' fbc:value='0'/>"
"      <fbc:fluxBound fbc:reaction='R00192' fbc:operation='lessEqual' fbc:value='999999'/>     "
"    </fbc:listOfFluxBounds>"
"    <fbc:listOfObjectives fbc:activeObjective='obj'>"
"      <fbc:objective fbc:id='obj' fbc:type='maximize'>"
"        <fbc:listOfFluxObjectives>"
"          <fbc:fluxObjective fbc:reaction='R00192' fbc:coefficient='1'/>"
"        </fbc:listOfFluxObjectives>"
"      </fbc:objective>"
"    </fbc:listOfObjectives>"
"  </model>"
"</sbml>";

  SBMLDocument* doc = readSBMLFromString(modelString);
  doc->checkConsistency();
  fail_unless(doc->getErrorLog()->contains(FbcSpeciesFormulaMustBeString));

  //convert to cobra just to test
  ConversionProperties prop;
  prop.addOption("convert fbc to cobra", true);
  fail_unless(doc->convert(prop) == LIBSBML_OPERATION_SUCCESS);
  
  // add species with notes 
  Species* testSpecies = doc->getModel()->createSpecies();
  testSpecies->initDefaults();
  testSpecies->setId("testSpecies");
  testSpecies->setNotes(
          "<body xmlns='http://www.w3.org/1999/xhtml'>"
          "  <p>FORMULA: Fe2S2X</p>                   "
          "  <p>CHARGE: -1</p>                        "
          "</body>                                    "
  );
  
  // convert it back
  prop.addOption("convert cobra", true);
  fail_unless(doc->convert(prop) == LIBSBML_OPERATION_SUCCESS);
  
  // check the formula
  Species* tested = doc->getModel()->getSpecies("testSpecies");
  fail_unless(tested != NULL);
  FbcSpeciesPlugin* fbcSpeciesPlugin = dynamic_cast<FbcSpeciesPlugin*>(tested->getPlugin("fbc"));
  fail_unless(fbcSpeciesPlugin != NULL);
  fail_unless(fbcSpeciesPlugin->getCharge() == -1);
  fail_unless(fbcSpeciesPlugin->getChemicalFormula() == "Fe2S2X");
  

}
END_TEST

START_TEST(test_FbcExtension_read_and_validate_chemicals)
{
  FbcPkgNamespaces *ns = new FbcPkgNamespaces();
  SBMLDocument*doc = new SBMLDocument(ns);
  doc->setPackageRequired("fbc", false);
  Model* model = doc->createModel();
  Compartment* comp = model->createCompartment();
  comp->initDefaults();
  comp->setId("comp");
  comp->setSize(1);
  Species *s = model->createSpecies();
  s->initDefaults();
  s->setId("s1");
  s->setCompartment("comp");
  s->setInitialAmount(1);
  FbcSpeciesPlugin* splugin = static_cast<FbcSpeciesPlugin*>(s->getPlugin("fbc"));
  fail_unless(splugin != NULL);
  
  // valid 
  splugin->setChemicalFormula("H2O");
  doc->checkInternalConsistency();
  fail_unless(!doc->getErrorLog()->contains(FbcSpeciesFormulaMustBeString));
  
  // valid
  doc->getErrorLog()->clearLog();
  splugin->setChemicalFormula("HO");
  doc->checkInternalConsistency();
  fail_unless(!doc->getErrorLog()->contains(FbcSpeciesFormulaMustBeString));

  // valid
  doc->getErrorLog()->clearLog();
  splugin->setChemicalFormula("");
  doc->checkInternalConsistency();
  fail_unless(!doc->getErrorLog()->contains(FbcSpeciesFormulaMustBeString));

  // invalid
  doc->getErrorLog()->clearLog();
  splugin->setChemicalFormula("hO");
  doc->checkInternalConsistency();
  fail_unless(doc->getErrorLog()->contains(FbcSpeciesFormulaMustBeString));

  // invalid
  doc->getErrorLog()->clearLog();
  splugin->setChemicalFormula("h1O");
  doc->checkInternalConsistency();
  fail_unless(doc->getErrorLog()->contains(FbcSpeciesFormulaMustBeString));

  // invalid
  doc->getErrorLog()->clearLog();
  splugin->setChemicalFormula("1hO");
  doc->checkInternalConsistency();
  fail_unless(doc->getErrorLog()->contains(FbcSpeciesFormulaMustBeString));

  // invalid
  doc->getErrorLog()->clearLog();
  splugin->setChemicalFormula("hO");
  doc->checkInternalConsistency();
  fail_unless(doc->getErrorLog()->contains(FbcSpeciesFormulaMustBeString));

  // invalid
  doc->getErrorLog()->clearLog();
  splugin->setChemicalFormula("_hO");
  doc->checkInternalConsistency();
  fail_unless(doc->getErrorLog()->contains(FbcSpeciesFormulaMustBeString));

  // invalid
  doc->getErrorLog()->clearLog();
  splugin->setChemicalFormula("H 2 O");
  doc->checkInternalConsistency();
  fail_unless(doc->getErrorLog()->contains(FbcSpeciesFormulaMustBeString));
  
  // invalid
  doc->getErrorLog()->clearLog();
  splugin->setChemicalFormula("H*_)(++2 O");
  doc->checkInternalConsistency();
  fail_unless(doc->getErrorLog()->contains(FbcSpeciesFormulaMustBeString));

  delete doc;
}
END_TEST

Suite *
create_suite_ReadFbcExtension (void)
{
  Suite *suite = suite_create("ReadFbcExtension");
  TCase *tcase = tcase_create("ReadFbcExtension");
  
  tcase_add_test( tcase, test_FbcExtension_OpertationTypes);
  tcase_add_test( tcase, test_FbcExtension_read_L3V1V1);
  tcase_add_test( tcase, test_FbcExtension_read_L3V1V1_defaultNS);
  tcase_add_test( tcase, test_FbcExtension_read_L3V1V1_unknown_elements);
  tcase_add_test( tcase, test_FbcExtension_read_L3V1V1_with_wonky_chemicals);
  tcase_add_test( tcase, test_FbcExtension_read_and_validate_chemicals);
  suite_add_tcase(suite, tcase);
  
  return suite;
}


CK_CPPEND
