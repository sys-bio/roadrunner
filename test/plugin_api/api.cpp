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


class PluginAPITests : public RoadRunnerTest {
public:
    path modelAnalysisModelsDir;

    PluginAPITests() {
        modelAnalysisModelsDir = rrTestModelsDir_ / "ModelAnalysis";
    }
};


TEST_F(PluginAPITests, PropertyList)
{
    string val = "val1";
    TELHandle para = tpCreateProperty("stringProp", "std::string", "", &val);
    PropertyBase* test = (PropertyBase*)tpCreateProperty("test", "listOfProperties", "", NULL);
    TELHandle properties = test->getValueHandle();
    tpAddPropertyToList(properties, para);
    string list = tpGetPropertyValueAsString(test);
    EXPECT_STREQ(list.c_str(), "[stringProp, val1]");
    TELHandle props = tpGetPropertyValueHandle(test);
}

