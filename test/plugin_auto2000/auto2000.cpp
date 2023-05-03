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


class PluginAuto2000Tests : public RoadRunnerTest {
public:
    path pluginsModelsDir;

    PluginAuto2000Tests() {
        pluginsModelsDir = rrTestModelsDir_ / "PLUGINS";
    }
};

//This just tests to make sure we don't crash if there's no boundary species.
PluginManager* PM = new PluginManager(rrPluginsBuildDir_.string());

TEST_F(PluginAuto2000Tests, Issue_773_no_boundary_species)
{
    Plugin* a2kplugin = PM->getPlugin("tel_auto2000");
    ASSERT_TRUE(a2kplugin != NULL);
    a2kplugin->setPropertyByString("SBML", (pluginsModelsDir / "auto2000_2rxn.xml").string().c_str());
    a2kplugin->setPropertyByString("PrincipalContinuationParameter", "k");
    a2kplugin->setPropertyByString("ScanDirection", "Positive");
    a2kplugin->setPropertyByString("RL0", "0");
    a2kplugin->setPropertyByString("RL1", "5");

    a2kplugin->execute();
}

delete PM;