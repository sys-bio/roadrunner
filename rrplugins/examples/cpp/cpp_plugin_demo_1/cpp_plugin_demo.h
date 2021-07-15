#ifndef cpp_plugin_demoH
#define cpp_plugin_demoH
#include "telCPPPlugin.h"
#include "telProperty.h"
//-------------------------------------------------------------------------

class TestPlugin : public tlp::CPPPlugin
{
    private:
        tlp::Property<int>         mTestProperty;

    public:
                                    TestPlugin();
                                   ~TestPlugin();
        
        bool                        execute(bool inThread);
};

extern "C"
{

TLP_DS  const char*     plugins_cc    getImplementationLanguage();
TLP_DS  TestPlugin*     plugins_cc    createPlugin();

// Plugin cleanup function
TLP_DS bool             plugins_cc    destroyPlugin(tlp::Plugin *plugin);

}

#endif
