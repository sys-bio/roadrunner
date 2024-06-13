#include "PluginMonteCarloTests.h"
#include "telPluginManager.h"
#include "telPlugin.h"
#include "telProperties.h"
#include "telTelluriumData.h"
#include "telProperty.h"
#include "../../wrappers/C/telplugins_properties_api.h"

using namespace tlp;

TEST_F(PluginMonteCarloTests, OPTIMIZE_TEST_MODEL)
{
    PluginManager* PM = new PluginManager(rrPluginsBuildDir_.string());

    Plugin* tmplugin = PM->getPlugin("tel_test_model");
    ASSERT_TRUE(tmplugin != NULL);
    tmplugin->execute();

    Plugin* mcplugin = PM->getPlugin("tel_monte_carlo_bs");
    ASSERT_TRUE(mcplugin != NULL);

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
    mcplugin->setPropertyByString("NrOfMCRuns", "100");
    mcplugin->setPropertyByString("FittedDataSelectionList", "[S1] [S2]");
    mcplugin->setPropertyByString("ExperimentalDataSelectionList", "[S1] [S2]");

    mcplugin->execute();
    //EXPECT_EQ(mcplugin->getPropertyValueAsString("StatusMessage").find("converged"), 0);

    TelluriumData* params = static_cast<TelluriumData*>(mcplugin->getPropertyValueHandle("MonteCarloParameters"));
    ASSERT_TRUE(params != NULL);
    EXPECT_EQ(params->rSize(), 100);
    EXPECT_EQ(params->cSize(), 1);
    for (int r = 0; r < params->rSize(); r++)
    {
        EXPECT_NEAR(params->getDataElement(r, 0), 1.0, 0.2);
    }

    Properties* means = static_cast<Properties*>(mcplugin->getPropertyValueHandle("Means"));
    ASSERT_TRUE(means != NULL);
    Property<double>* mean1 = static_cast<Property<double>*>(means->getFirst());
    EXPECT_EQ(mean1->getName(), "k1");
    EXPECT_NEAR(mean1->getValue(), 1.0, 0.1);

    Properties* conf_intervals = static_cast<Properties*>(mcplugin->getPropertyValueHandle("ConfidenceIntervals"));
    ASSERT_TRUE(conf_intervals != NULL);
    Property<double>* ci = static_cast<Property<double>* >(conf_intervals->getFirst());
    EXPECT_EQ(ci->getName(), "k1");
    EXPECT_NEAR(ci->getValue(), 0.05, 0.03);

    Properties* percentiles = static_cast<Properties*>(mcplugin->getPropertyValueHandle("Percentiles"));
    ASSERT_TRUE(percentiles != NULL);
    Property<double>* percentile = static_cast<Property<double>*>(percentiles->getFirst());
    ASSERT_TRUE(percentile != NULL);
    EXPECT_EQ(percentile->getName(), "k1_25_percentile");
    EXPECT_NEAR(percentile->getValue(), 0.99, 0.12);

    percentile = static_cast<Property<double>*>(percentiles->getNext());
    ASSERT_TRUE(percentile != NULL);
    EXPECT_EQ(percentile->getName(), "k1_75_percentile");
    EXPECT_NEAR(percentile->getValue(), 1.01, 0.12);

    percentile = static_cast<Property<double>*>(percentiles->getNext());
    ASSERT_TRUE(percentile != NULL);
    EXPECT_EQ(percentile->getName(), "k1_02.5_percentile");
    EXPECT_NEAR(percentile->getValue(), 0.96, 0.13);

    percentile = static_cast<Property<double>*>(percentiles->getNext());
    ASSERT_TRUE(percentile != NULL);
    EXPECT_EQ(percentile->getName(), "k1_97.5_percentile");
    EXPECT_NEAR(percentile->getValue(), 1.04, 0.13);

    percentile = static_cast<Property<double>*>(percentiles->getNext());
    ASSERT_TRUE(percentile != NULL);
    EXPECT_EQ(percentile->getName(), "k1_01_percentile");
    EXPECT_NEAR(percentile->getValue(), 0.95, 0.14);

    percentile = static_cast<Property<double>*>(percentiles->getNext());
    ASSERT_TRUE(percentile != NULL);
    EXPECT_EQ(percentile->getName(), "k1_99_percentile");
    EXPECT_NEAR(percentile->getValue(), 1.05, 0.14);

    EXPECT_TRUE(percentiles->getNext() == NULL);
}