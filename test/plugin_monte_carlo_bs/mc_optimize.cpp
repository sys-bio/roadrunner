#include "gtest/gtest.h"
#include "../test_util.h"
#include "telPluginManager.h"
#include "telPlugin.h"
#include "telProperties.h"
#include "telTelluriumData.h"
#include "telProperty.h"
//#include "tel"


using namespace testing;
using namespace std;
using namespace tlp;

extern string gRRTestDir;
extern string gRROutputDir;
extern string gRRPluginDir;
//extern APIHandleManager gHM;


TEST(RRPLUGIN_TEST_MODEL, OPTIMIZE_TEST_MODEL)
{
    if (gRRPluginDir.empty()) {
        std::cerr << "Please set the 'plugindir' environment variable before running the plugin tests.  This should be the directory where the plugin dlls are created." << std::endl;
        EXPECT_TRUE(false);
    }
    PluginManager* PM = new PluginManager(gRRPluginDir);

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

TEST(RRPLUGIN_TEST_MODEL, OPTIMIZE_NELDER_MEAD)
{
    if (gRRPluginDir.empty()) {
        std::cerr << "Please set the 'plugindir' environment variable before running the plugin tests.  This should be the directory where the plugin dlls are created." << std::endl;
        EXPECT_TRUE(false);
    }
    PluginManager* PM = new PluginManager(gRRPluginDir);

    Plugin* tmplugin = PM->getPlugin("tel_test_model");
    ASSERT_TRUE(tmplugin != NULL);
    tmplugin->execute();

    Plugin* mcplugin = PM->getPlugin("tel_monte_carlo_bs");
    ASSERT_TRUE(mcplugin != NULL);

    mcplugin->setPropertyByString("MinimizerPlugin", "tel_nelder_mead");

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
        EXPECT_NEAR(params->getDataElement(r, 0), 1.0, 0.5);
    }

    Properties* means = static_cast<Properties*>(mcplugin->getPropertyValueHandle("Means"));
    ASSERT_TRUE(means != NULL);
    Property<double>* mean1 = static_cast<Property<double>*>(means->getFirst());
    EXPECT_EQ(mean1->getName(), "k1");
    EXPECT_NEAR(mean1->getValue(), 1.0, 0.4);

    Properties* conf_intervals = static_cast<Properties*>(mcplugin->getPropertyValueHandle("ConfidenceIntervals"));
    ASSERT_TRUE(conf_intervals != NULL);
    Property<double>* ci = static_cast<Property<double>*>(conf_intervals->getFirst());
    EXPECT_EQ(ci->getName(), "k1");
    EXPECT_NEAR(ci->getValue(), 0.2, 0.2);

    Properties* percentiles = static_cast<Properties*>(mcplugin->getPropertyValueHandle("Percentiles"));
    ASSERT_TRUE(percentiles != NULL);
    Property<double>* percentile = static_cast<Property<double>*>(percentiles->getFirst());
    ASSERT_TRUE(percentile != NULL);
    EXPECT_EQ(percentile->getName(), "k1_25_percentile");
    EXPECT_NEAR(percentile->getValue(), 1.0, 0.25);

    percentile = static_cast<Property<double>*>(percentiles->getNext());
    ASSERT_TRUE(percentile != NULL);
    EXPECT_EQ(percentile->getName(), "k1_75_percentile");
    EXPECT_NEAR(percentile->getValue(), 1.03, 0.3);

    percentile = static_cast<Property<double>*>(percentiles->getNext());
    ASSERT_TRUE(percentile != NULL);
    EXPECT_EQ(percentile->getName(), "k1_02.5_percentile");
    EXPECT_NEAR(percentile->getValue(), 0.875, 0.32);

    percentile = static_cast<Property<double>*>(percentiles->getNext());
    ASSERT_TRUE(percentile != NULL);
    EXPECT_EQ(percentile->getName(), "k1_97.5_percentile");
    EXPECT_NEAR(percentile->getValue(), 1.09, 0.4);

    percentile = static_cast<Property<double>*>(percentiles->getNext());
    ASSERT_TRUE(percentile != NULL);
    EXPECT_EQ(percentile->getName(), "k1_01_percentile");
    EXPECT_NEAR(percentile->getValue(), 0.875, 0.4);

    percentile = static_cast<Property<double>*>(percentiles->getNext());
    ASSERT_TRUE(percentile != NULL);
    EXPECT_EQ(percentile->getName(), "k1_99_percentile");
    EXPECT_NEAR(percentile->getValue(), 1.09, 0.4);

    EXPECT_TRUE(percentiles->getNext() == NULL);
}

TEST(RRPLUGIN_TEST_MODEL, CHECK_SEED)
{
    if (gRRPluginDir.empty()) {
        std::cerr << "Please set the 'plugindir' environment variable before running the plugin tests.  This should be the directory where the plugin dlls are created." << std::endl;
        EXPECT_TRUE(false);
    }
    PluginManager* PM = new PluginManager(gRRPluginDir);

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

