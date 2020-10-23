#ifndef TestModelH
#define TestModelH
#include "telProperty.h"
#include "rrplugins/pluginBaseClass/telCPPPlugin.h"
#include "telTelluriumData.h"
#include "rrplugins/core/tel_api.h"
//---------------------------------------------------------------------------


namespace testModel {

    using namespace tlp;

    class TestModel : public CPPPlugin
    {
    public:
        TestModel();
        ~TestModel();
        bool                        execute(bool);
        unsigned char*              getManualAsPDF() const;
        size_t                      getPDFManualByteSize();

    private:
        Property<string>            mModel;
        Property<TelluriumData>     mTestData;
        Property<TelluriumData>     mTestDataWithNoise;
        Property<double>            mSigma;
        void                        addWeights();
        void                        assignPropertyDescriptions();
    };

    #ifdef EXPORT_TEST_MODEL
    extern "C"
    {
        TLP_DS Plugin* plugins_cc createPlugin();
        TLP_DS const char* plugins_cc getImplementationLanguage();
        TLP_DS void plugins_cc setHostInterface(rrc::THostInterface* _hostInterface);
    }
    #endif
}

#endif
