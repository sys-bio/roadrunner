#include "gtest/gtest.h"
#include "../test_util.h"
#include "telPluginManager.h"
#include "telPlugin.h"
#include "telProperties.h"
#include "telTelluriumData.h"
#include "telProperty.h"
#include "../../wrappers/C/telplugins_properties_api.h"


using namespace testing;
using namespace std;
using namespace tlp;

extern string gRRTestDir;
extern string gRROutputDir;
extern string gRRPluginDir;
//extern APIHandleManager gHM;


TEST(RRPLUGIN_API_TEST, PROPERTY_LIST)
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

