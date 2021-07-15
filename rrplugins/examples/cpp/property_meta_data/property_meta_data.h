#ifndef property_meta_dataH
#define property_meta_dataH
#include "../source/telCPPPlugin.h"
#include "../source/telProperty.h"
//---------------------------------------------------------------------------


class TestPlugin : public tlp::CPPPlugin
{
    private:
        tlp::Property<int>          mTestProperty;

    public:
                                    TestPlugin(int testParameter = 0);
                                   ~TestPlugin();

        bool                        execute(bool inThread = false);
};

extern "C"
{
TLP_DS  const char*     plugins_cc    getImplementationLanguage();
TLP_DS  TestPlugin*     plugins_cc    createPlugin(rr::RoadRunner* aRR);

// Plugin cleanup function
TLP_DS bool             plugins_cc    destroyPlugin(tlp::Plugin *plugin);

}

#endif
