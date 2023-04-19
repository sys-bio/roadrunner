#include "PluginAuto2000Tests.h"
#include "telPluginManager.h"
#include "telPlugin.h"
#include "telProperties.h"
#include "telTelluriumData.h"
#include "telProperty.h"
#include "../../wrappers/C/telplugins_properties_api.h"

using namespace tlp;

TEST_F(PluginAuto2000Tests, RUN_BISTABLE_IRREVERSIBLE)
{
PluginManager* PM = new PluginManager(rrPluginsBuildDir_.string());
/*
Plugin* a2kplugin = PM->getPlugin("tel_auto2000");
ASSERT_TRUE(a2kplugin != NULL);

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
 */
}