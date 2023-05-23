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

using namespace testing;
using namespace std;
using namespace tlp;


class PluginTests : public RoadRunnerTest {
public:
    path pluginsModelsDir;

    PluginTests() {
        pluginsModelsDir = rrTestModelsDir_ / "PLUGINS";
    }

    static void SetUpTestSuite() {
        if (PM == NULL)
            PM = new PluginManager(getRoadRunnerPluginBuildDirectory().string());
    }

protected:
    static PluginManager* PM;
};

PluginManager* PluginTests::PM = NULL;


TEST_F(PluginTests, AUTO_2000_Issue_773_no_boundary_species)
{
    Plugin* a2kplugin = PM->getPlugin("tel_auto2000");
    ASSERT_TRUE(a2kplugin != NULL);

    // reset the value of plugin properties
    a2kplugin->resetPropertiesValues();

    a2kplugin->setPropertyByString("SBML", (pluginsModelsDir / "auto2000_2rxn.xml").string().c_str());
    a2kplugin->setPropertyByString("PrincipalContinuationParameter", "k");
    a2kplugin->setPropertyByString("ScanDirection", "Positive");
    a2kplugin->setPropertyByString("RL0", "0");
    a2kplugin->setPropertyByString("RL1", "5");

    a2kplugin->execute();
}

TEST_F(PluginTests, AUTO_2000_RUN_BIOMOD_203)
{
    Plugin* a2kplugin = PM->getPlugin("tel_auto2000");
    ASSERT_TRUE(a2kplugin != NULL);

    // reset the value of plugin properties
    a2kplugin->resetPropertiesValues();

    a2kplugin->setPropertyByString("SBML", (pluginsModelsDir / "BIOMD0000000203.xml").string().c_str());
    a2kplugin->setPropertyByString("ScanDirection", "Positive");
    a2kplugin->setPropertyByString("PrincipalContinuationParameter", "A");
    a2kplugin->setPropertyByString("PCPLowerBound", "10");
    a2kplugin->setPropertyByString("PCPUpperBound", "200");
    a2kplugin->setPropertyByString("NMX", "5000");

    a2kplugin->execute();

    string summary = a2kplugin->getPropertyValueAsString("BifurcationSummary");
    string headers = "BR    PT  TY LAB    PAR(0)        L2-NORM         U(1)          U(2)          U(3)          U(4)          U(5)";
    EXPECT_EQ(summary.find(headers), 4);

    vector<int>* points = (vector<int>*)a2kplugin->getPropertyValueHandle("BifurcationPoints");
    ASSERT_TRUE(points != NULL);
    ASSERT_EQ(points->size(), 4);
    EXPECT_EQ(points->at(0), 1);
    EXPECT_EQ(points->at(1), 1385);
    EXPECT_EQ(points->at(2), 2189);
    EXPECT_EQ(points->at(3), 3705);

    StringList* labels = (StringList*)a2kplugin->getPropertyValueHandle("BifurcationLabels");
    ASSERT_TRUE(labels != NULL);
    ASSERT_EQ(labels->size(), 4);
    EXPECT_EQ(labels->asString(), "EP,LP,LP,EP");

    TelluriumData* data = (TelluriumData*)a2kplugin->getPropertyValueHandle("BifurcationData");
    ASSERT_TRUE(data != NULL);
    EXPECT_EQ(data->cSize(), 6);
    EXPECT_EQ(data->rSize(), 3705);
    EXPECT_EQ(data->getColumnNamesAsString(), "A,[OCT4],[SOX2],[NANOG],[Protein],[OCT4_SOX2]");
    //Spot checks:
    EXPECT_NEAR(data->getDataElement(6, 0), 10.036, 0.001);
    EXPECT_NEAR(data->getDataElement(52, 1), 8.85978, 0.0001);
    EXPECT_NEAR(data->getDataElement(167, 2), 12.3176, 0.0001);
    EXPECT_NEAR(data->getDataElement(623, 3), 0.0469995, 0.0001);
    EXPECT_NEAR(data->getDataElement(1522, 4), 33.0702, 0.0001);
    EXPECT_NEAR(data->getDataElement(2345, 5), 22.6297, 0.0001);
    EXPECT_NEAR(data->getDataElement(3535, 6), 183.378, 0.001);
}

/*
TEST_F(PluginTests, AUTO_2000_RUN_BISTABLE)
{
    Plugin* a2kplugin = PM->getPlugin("tel_auto2000");
    ASSERT_TRUE(a2kplugin != NULL);

    // reset the value of plugin properties
    a2kplugin->resetPropertiesValues();

    a2kplugin->setPropertyByString("SBML", (pluginsModelsDir / "bistable.xml").string().c_str());
    a2kplugin->setPropertyByString("ScanDirection", "Negative");
    a2kplugin->setPropertyByString("PrincipalContinuationParameter", "k3");
    a2kplugin->setPropertyByString("PCPLowerBound", "0.35");
    a2kplugin->setPropertyByString("PCPUpperBound", "1.5");
    a2kplugin->setPropertyByString("NMX", "5000");

    a2kplugin->execute();

    string summary = a2kplugin->getPropertyValueAsString("BifurcationSummary");
    string headers = "BR    PT  TY LAB    PAR(0)        L2-NORM         U(1)";
    EXPECT_EQ(summary.find(headers), 4);

    vector<int>* points = (vector<int>*)a2kplugin->getPropertyValueHandle("BifurcationPoints");
    ASSERT_TRUE(points != NULL);
    ASSERT_EQ(points->size(), 4);
    EXPECT_EQ(points->at(0), 1);
    EXPECT_EQ(points->at(1), 47);
    EXPECT_EQ(points->at(2), 67);
    EXPECT_EQ(points->at(3), 97);

    StringList* labels = (StringList*)a2kplugin->getPropertyValueHandle("BifurcationLabels");
    ASSERT_TRUE(labels != NULL);
    ASSERT_EQ(labels->size(), 4);
    EXPECT_EQ(labels->asString(), "EP,LP,LP,EP");

    TelluriumData* data = (TelluriumData*)a2kplugin->getPropertyValueHandle("BifurcationData");
    ASSERT_TRUE(data != NULL);
    EXPECT_EQ(data->cSize(), 2);
    EXPECT_EQ(data->rSize(), 97);
    EXPECT_EQ(data->getColumnNamesAsString(), "k3,[x]");
    //Spot checks:
    EXPECT_NEAR(data->getDataElement(17, 0), 1.16386, 0.0001);
    EXPECT_NEAR(data->getDataElement(93, 1), 2.63297, 0.0001);
}
 */

TEST_F(PluginTests, AUTO_2000_RUN_BISTABLE_IRREVERSIBLE)
{
    Plugin* a2kplugin = PM->getPlugin("tel_auto2000");
    ASSERT_TRUE(a2kplugin != NULL);

    // reset the value of plugin properties
    a2kplugin->resetPropertiesValues();

    a2kplugin->setPropertyByString("SBML", (pluginsModelsDir / "irreversible_bistability.xml").string().c_str());
    a2kplugin->setPropertyByString("ScanDirection", "Positive");
    a2kplugin->setPropertyByString("PrincipalContinuationParameter", "Signal");
    a2kplugin->setPropertyByString("PCPLowerBound", "-3");
    a2kplugin->setPropertyByString("PCPUpperBound", "4");
    a2kplugin->setPropertyByString("PreSimulation", "True");
    a2kplugin->setPropertyByString("NMX", "5000");

    a2kplugin->execute();

    string summary = a2kplugin->getPropertyValueAsString("BifurcationSummary");
    string headers = "BR    PT  TY LAB    PAR(0)        L2-NORM         U(1)          U(2)";
    EXPECT_EQ(summary.find(headers), 4);

    vector<int>* points = (vector<int>*)a2kplugin->getPropertyValueHandle("BifurcationPoints");
    ASSERT_TRUE(points != NULL);
    ASSERT_EQ(points->size(), 4);
    EXPECT_EQ(points->at(0), 1);
    EXPECT_EQ(points->at(1), 116);
    EXPECT_EQ(points->at(2), 255);
    EXPECT_EQ(points->at(3), 361);

    StringList* labels = (StringList*)a2kplugin->getPropertyValueHandle("BifurcationLabels");
    ASSERT_TRUE(labels != NULL);
    ASSERT_EQ(labels->size(), 4);
    EXPECT_EQ(labels->asString(), "EP,LP,LP,EP");

    TelluriumData* data = (TelluriumData*)a2kplugin->getPropertyValueHandle("BifurcationData");
    ASSERT_TRUE(data != NULL);
    EXPECT_EQ(data->cSize(), 3);
    EXPECT_EQ(data->rSize(), 361);
    EXPECT_EQ(data->getColumnNamesAsString(), "Signal,[R1],[EP]");
    //Spot checks:
    EXPECT_NEAR(data->getDataElement(17, 0), -2.39214, 0.0001);
    EXPECT_NEAR(data->getDataElement(93, 1), 3.0908, 0.0001);
    EXPECT_NEAR(data->getDataElement(193, 2), 10.5904, 0.0001);
}

TEST_F(PluginTests, TEST_MODEL_STANDARD_RUN)
{
    Plugin* tmplugin = PM->getPlugin("tel_test_model");
    ASSERT_TRUE(tmplugin != NULL);

    // reset the value of plugin properties
    tmplugin->resetPropertiesValues();

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

TEST_F(PluginTests, TEST_MODEL_CHECK_SEED)
{
    Plugin* tmplugin = PM->getPlugin("tel_test_model");
    ASSERT_TRUE(tmplugin != NULL);

    // reset the value of plugin properties
    tmplugin->resetPropertiesValues();

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
}

/*
TEST_F(PluginTests, TEST_MODEL_CHECK_SIGMA)
{
    Plugin* tmplugin = PM->getPlugin("tel_test_model");
    ASSERT_TRUE(tmplugin != NULL);

    // reset the value of plugin properties
    tmplugin->resetPropertiesValues();

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
}
*/

TEST_F(PluginTests, TEST_MODEL_NEW_MODEL)
{
    Plugin* tmplugin = PM->getPlugin("tel_test_model");
    ASSERT_TRUE(tmplugin != NULL);

    // reset the value of plugin properties
    tmplugin->resetPropertiesValues();

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