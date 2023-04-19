#include "PluginAuto2000Tests.h"
#include "telPluginManager.h"
#include "telPlugin.h"
#include "telProperties.h"
#include "telTelluriumData.h"
#include "telProperty.h"
#include "../../wrappers/C/telplugins_properties_api.h"

using namespace tlp;

TEST_F(PluginAuto2000Tests, RUN_BISTABLE)
{
PluginManager* PM = new PluginManager(rrPluginsBuildDir_.string());
/*
Plugin* a2kplugin = PM->getPlugin("tel_auto2000");
ASSERT_TRUE(a2kplugin != NULL);

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
 */
delete PM;
}