#include "gtest/gtest.h"
#include <filesystem>
#include "RoadRunnerTest.h"
#include "telPluginManager.h"
#include "telPlugin.h"
#include "telProperties.h"
#include "telTelluriumData.h"
#include "telProperty.h"
#include "../../wrappers/C/telplugins_properties_api.h"

using std::filesystem::path;

using namespace tlp;

class PluginTestModelTests : public RoadRunnerTest {
public:
    path pluginsModelsDir;

    PluginTestModelTests() {
        pluginsModelsDir = rrTestModelsDir_ / "PLUGINS";
    }
};


TEST_F(PluginTestModelTests, STANDARD_RUN)
{
    PluginManager* PM = new PluginManager(rrPluginsBuildDir_.string());

    Plugin* tmplugin = PM->getPlugin("tel_test_model");
    ASSERT_TRUE(tmplugin != NULL);
    tmplugin->execute();

    PropertyBase* sbml = tmplugin->getProperty("Model");
    EXPECT_TRUE(sbml->getValueAsString().find("<sbml") != string::npos);
    EXPECT_TRUE(sbml->getValueAsString().find("k1") != string::npos);

    PropertyBase* noisedata = tmplugin->getProperty("TestDataWithNoise");
    TelluriumData* noise = static_cast<TelluriumData*>(noisedata->getValueHandle());
    EXPECT_EQ(noise->cSize(), 3);
    EXPECT_EQ(noise->rSize(), 14);

    PropertyBase* testdata = tmplugin->getProperty("TestData");
    TelluriumData* sim = static_cast<TelluriumData*>(testdata->getValueHandle());
    EXPECT_EQ(sim->cSize(), 3);
    EXPECT_EQ(sim->rSize(), 14);
    
    double sumdiff = 0;
    for (int r = 0; r < sim->rSize(); r++)
    {
        //The 'time' column should be identical:
        EXPECT_EQ(sim->getDataElement(r, 0), noise->getDataElement(r, 0));

        for (int c = 1; c < sim->cSize(); c++)
        {
            EXPECT_NE(sim->getDataElement(r, c), noise->getDataElement(r, c));
            sumdiff += abs(sim->getDataElement(r, c) - noise->getDataElement(r, c));
        }
    }
    EXPECT_NEAR(sumdiff, 3.e-6*28, 1e-4);

    PropertyBase* sig = tmplugin->getProperty("Sigma");
    double* sigma = static_cast<double*>(sig->getValueHandle());
    EXPECT_EQ(*sigma, 3.e-6);

    PropertyBase* seedprop = tmplugin->getProperty("Seed");
    unsigned long* seed = static_cast<unsigned long*>(seedprop->getValueHandle());
    EXPECT_EQ(*seed, 0);
}

TEST_F(PluginTestModelTests, CHECK_SEED)
{
    /*
    PluginManager* PM = new PluginManager(rrPluginsBuildDir_.string());

    Plugin* tmplugin = PM->getPlugin("tel_test_model");
    ASSERT_TRUE(tmplugin != NULL);

    tmplugin->setPropertyByString("Seed", "1001");
    tmplugin->execute();

    PropertyBase* seedprop = tmplugin->getProperty("Seed");
    unsigned long* seed = static_cast<unsigned long*>(seedprop->getValueHandle());
    EXPECT_EQ(*seed, 1001);

    PropertyBase* noisedata = tmplugin->getProperty("TestDataWithNoise");
    TelluriumData* noise = static_cast<TelluriumData*>(noisedata->getValueHandle());
    EXPECT_EQ(noise->cSize(), 3);
    EXPECT_EQ(noise->rSize(), 14);

    TelluriumData s1001a(*noise);

    tmplugin->execute();

    TelluriumData s1001b(*noise);

    tmplugin->setPropertyByString("Seed", "1004");
    tmplugin->execute();

    TelluriumData s1004(*noise);

    double sumdiff = 0;
    for (int r = 0; r < s1001a.rSize(); r++)
    {
        //The 'time' column should be identical:
        EXPECT_EQ(s1001a.getDataElement(r, 0), s1001b.getDataElement(r, 0));
        EXPECT_EQ(s1001a.getDataElement(r, 0), s1004.getDataElement(r, 0));

        for (int c = 1; c < s1001a.cSize(); c++)
        {
            EXPECT_EQ(s1001a.getDataElement(r, c), s1001b.getDataElement(r, c));
            sumdiff += abs(s1001a.getDataElement(r, c) - s1004.getDataElement(r, c));
        }
    }
    EXPECT_NEAR(sumdiff, 3.e-6 * 2 * 28, 1e-4);
    EXPECT_GT(sumdiff, 0);
    */
}

TEST_F(PluginTestModelTests, CHECK_SIGMA)
{
    /*
    PluginManager* PM = new PluginManager(rrPluginsBuildDir_.string());

    Plugin* tmplugin = PM->getPlugin("tel_test_model");
    ASSERT_TRUE(tmplugin != NULL);

    tmplugin->execute();

    PropertyBase* testdata = tmplugin->getProperty("TestData");
    TelluriumData* sim = static_cast<TelluriumData*>(testdata->getValueHandle());

    PropertyBase* noisedata = tmplugin->getProperty("TestDataWithNoise");
    TelluriumData* noise = static_cast<TelluriumData*>(noisedata->getValueHandle());
    EXPECT_EQ(noise->cSize(), 3);
    EXPECT_EQ(noise->rSize(), 14);

    TelluriumData siglow(*noise);

    tmplugin->setPropertyByString("Sigma", "10");
    tmplugin->execute();

    TelluriumData sighigh(*noise);

    double sumdiff_low = 0, sumdiff_high = 0;
    for (int r = 0; r < siglow.rSize(); r++)
    {
        for (int c = 1; c < siglow.cSize(); c++)
        {
            sumdiff_low  += abs(siglow.getDataElement(r, c)  - sim->getDataElement(r, c));
            sumdiff_high += abs(sighigh.getDataElement(r, c) - sim->getDataElement(r, c));
        }
    }
    EXPECT_NEAR(sumdiff_low, 3.e-6 * 28, 1e-4);
    EXPECT_NEAR(sumdiff_high, 10 * 28, 200);

    PropertyBase* sig = tmplugin->getProperty("Sigma");
    double* sigma = static_cast<double*>(sig->getValueHandle());
    EXPECT_EQ(*sigma, 10);
    */
}

TEST_F(PluginTestModelTests, NEW_MODEL)
{
    /*
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
    */
}

