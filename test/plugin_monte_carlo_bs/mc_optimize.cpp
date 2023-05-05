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

class PluginMonteCarloTests : public RoadRunnerTest {
public:
    path pluginsModelsDir;

    PluginMonteCarloTests() {
        pluginsModelsDir = rrTestModelsDir_ / "PLUGINS";
    }
};


TEST_F(PluginMonteCarloTests, OPTIMIZE_NELDER_MEAD)
{

    PluginManager* PM = new PluginManager(rrPluginsBuildDir_.string());

    Plugin* tmplugin = PM->getPlugin("tel_test_model");
    ASSERT_TRUE(tmplugin != NULL);
    tmplugin->execute();

    Plugin* mcplugin = PM->getPlugin("tel_monte_carlo_bs");
    ASSERT_TRUE(mcplugin != NULL);

    mcplugin->setPropertyByString("Seed", "2001");

    PropertyBase* sbml = tmplugin->getProperty("Model");
    mcplugin->setPropertyByString("SBML", sbml->getValueAsString().c_str());

    PropertyBase* testdata = tmplugin->getProperty("TestDataWithNoise");
    TelluriumData* exdata = static_cast<TelluriumData*>(testdata->getValueHandle());
    mcplugin->setPropertyValue("ExperimentalData", exdata);

    Property<double> k1val(0.3, "k1", "", "", "", true);
    Properties ipl;
    ipl.add(&k1val);
    //tlp::Property tpcre();
    mcplugin->setPropertyValue("InputParameterList", &ipl);
    mcplugin->setPropertyByString("NrOfMCRuns", "40");
    mcplugin->setPropertyByString("FittedDataSelectionList", "[S1] [S2]");
    mcplugin->setPropertyByString("ExperimentalDataSelectionList", "[S1] [S2]");

    mcplugin->execute();
    //EXPECT_EQ(mcplugin->getPropertyValueAsString("StatusMessage").find("converged"), 0);

    TelluriumData* params = static_cast<TelluriumData*>(mcplugin->getPropertyValueHandle("MonteCarloParameters"));
    ASSERT_TRUE(params != NULL);
    TelluriumData copy(*params);

    Properties* conf_intervals = static_cast<Properties*>(mcplugin->getPropertyValueHandle("ConfidenceIntervals"));
    ASSERT_TRUE(conf_intervals != NULL);
    Property<double>* cl = static_cast<Property<double>*>(conf_intervals->getFirst());
    ASSERT_TRUE(cl != NULL);
    double cl_one = cl->getValue();

    mcplugin->execute();
    params = static_cast<TelluriumData*>(mcplugin->getPropertyValueHandle("MonteCarloParameters"));
    ASSERT_TRUE(params != NULL);

    EXPECT_EQ(params->rSize(), 40);
    EXPECT_EQ(params->cSize(), 1);
    for (int r = 0; r < params->rSize(); r++)
    {
        EXPECT_EQ(params->getDataElement(r, 0), copy.getDataElement(r, 0));
    }

    conf_intervals = static_cast<Properties*>(mcplugin->getPropertyValueHandle("ConfidenceIntervals"));
    ASSERT_TRUE(conf_intervals != NULL);
    cl = static_cast<Property<double>*>(conf_intervals->getFirst());
    ASSERT_TRUE(cl != NULL);

    EXPECT_EQ(cl_one, cl->getValue());
}
