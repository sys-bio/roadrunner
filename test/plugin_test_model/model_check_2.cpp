#include "PluginTestModelTests.h"
#include "telPluginManager.h"
#include "telPlugin.h"
#include "telProperties.h"
#include "telTelluriumData.h"
#include "telProperty.h"
#include "../../wrappers/C/telplugins_properties_api.h"

using namespace tlp;

TEST_F(PluginTestModelTests, CHECK_SEED)
{
PluginManager* PM = new PluginManager(rrPluginsBuildDir_.string());
/*

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
delete PM;
}