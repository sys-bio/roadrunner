#include "PluginTestModelTests.h"
#include "telPluginManager.h"
#include "telPlugin.h"
#include "telProperties.h"
#include "telTelluriumData.h"
#include "telProperty.h"
#include "../../wrappers/C/telplugins_properties_api.h"

using namespace tlp;

TEST_F(PluginTestModelTests, NEW_MODEL)
{
    PluginManager* PM = new PluginManager(rrPluginsBuildDir_.string());

    Plugin* tmplugin = PM->getPlugin("tel_test_model");
    ASSERT_TRUE(tmplugin != NULL);

    string  newModel = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
    <sbml xmlns=\"http://www.sbml.org/sbml/level3/version2/core\" level=\"3\" version=\"2\">\n\
    <model metaid=\"_case00001\" id=\"case00001\" name=\"case00001\">\n\
    <listOfCompartments>\n\
      <compartment id=\"compartment\" name=\"compartment\" size=\"1\" constant=\"true\"/>\n\
    </listOfCompartments>\n\
    <listOfSpecies>\n\
      <species id=\"S1\" name=\"S1\" compartment=\"compartment\" initialAmount=\"0.00015\"  hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n\
      <species id=\"S2\" name=\"S2\" compartment=\"compartment\" initialAmount=\"0\"  hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n\
      <species id=\"S3\" name=\"S3\" compartment=\"compartment\" initialAmount=\"0\" hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n\
    </listOfSpecies>\n\
    <listOfParameters>\n\
      <parameter id=\"k1\" name=\"k1\" value=\"1\" constant=\"true\"/>\n\
      <parameter id=\"k2\" name=\"k1\" value=\"0.5\" constant=\"true\"/>\n\
    </listOfParameters>\n\
    <listOfReactions>\n\
      <reaction id=\"reaction1\" name=\"reaction1\" reversible=\"false\">\n\
        <listOfReactants>\n\
          <speciesReference species=\"S1\" stoichiometry=\"1\" constant=\"true\"/>\n\
        </listOfReactants>\n\
        <listOfProducts>\n\
          <speciesReference species=\"S2\" stoichiometry=\"1\" constant=\"true\"/>\n\
        </listOfProducts>\n\
        <kineticLaw>\n\
          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n\
            <apply>\n\
              <times/>\n\
              <ci> compartment </ci>\n\
              <ci> k1 </ci>\n\
              <ci> S1 </ci>\n\
            </apply>\n\
          </math>\n\
        </kineticLaw>\n\
      </reaction>\n\
      <reaction id=\"reaction2\" reversible=\"false\">\n\
        <listOfReactants>\n\
          <speciesReference species=\"S2\" stoichiometry=\"1\" constant=\"true\"/>\n\
        </listOfReactants>\n\
        <listOfProducts>\n\
          <speciesReference species=\"S3\" stoichiometry=\"1\" constant=\"true\"/>\n\
        </listOfProducts>\n\
        <kineticLaw>\n\
          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n\
            <apply>\n\
              <times/>\n\
              <ci> compartment </ci>\n\
              <ci> k2 </ci>\n\
              <ci> S2 </ci>\n\
            </apply>\n\
          </math>\n\
        </kineticLaw>\n\
      </reaction>\n\
    </listOfReactions>\n\
    </model>\n\
    </sbml>\n\
    ";

    tmplugin->setPropertyByString("Model", newModel.c_str());
    tmplugin->execute();

    PropertyBase* sbml = tmplugin->getProperty("Model");
    EXPECT_EQ(sbml->getValueAsString(), newModel);

    PropertyBase* noisedata = tmplugin->getProperty("TestDataWithNoise");
    TelluriumData* noise = static_cast<TelluriumData*>(noisedata->getValueHandle());
    EXPECT_EQ(noise->cSize(), 4);
    EXPECT_EQ(noise->rSize(), 14);

    PropertyBase* testdata = tmplugin->getProperty("TestData");
    TelluriumData* sim = static_cast<TelluriumData*>(testdata->getValueHandle());
    EXPECT_EQ(sim->cSize(), 4);
    EXPECT_EQ(sim->rSize(), 14);

}

