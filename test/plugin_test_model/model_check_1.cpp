#include "PluginTestModelTests.h"
#include "telPluginManager.h"
#include "telPlugin.h"
#include "telProperties.h"
#include "telTelluriumData.h"
#include "telProperty.h"
#include "../../wrappers/C/telplugins_properties_api.h"

using namespace tlp;

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

    PM->unloadAll();
}