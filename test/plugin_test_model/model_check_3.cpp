#include "PluginTestModelTests.h"
#include "telPluginManager.h"
#include "telPlugin.h"
#include "telProperties.h"
#include "telTelluriumData.h"
#include "telProperty.h"
#include "../../wrappers/C/telplugins_properties_api.h"

using namespace tlp;

TEST_F(PluginTestModelTests, CHECK_SIGMA)
{
PluginManager* PM = new PluginManager(rrPluginsBuildDir_.string());
/*
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
delete PM;
}