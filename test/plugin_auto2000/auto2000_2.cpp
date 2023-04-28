#include "PluginAuto2000Tests.h"
#include "telPluginManager.h"
#include "telPlugin.h"
#include "telProperties.h"
#include "telTelluriumData.h"
#include "telProperty.h"
#include "../../wrappers/C/telplugins_properties_api.h"

using namespace tlp;

TEST_F(PluginAuto2000Tests, RUN_BIOMOD_203)
{
    PluginManager* PM = new PluginManager(rrPluginsBuildDir_.string());
    Plugin* a2kplugin = PM->getPlugin("tel_auto2000");
    ASSERT_TRUE(a2kplugin != NULL);

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