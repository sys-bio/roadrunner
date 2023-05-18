#ifndef TestModelH
#define TestModelH
#include "telProperty.h"
#include "rrplugins/pluginBaseClass/telCPPPlugin.h"
#include "telTelluriumData.h"
#include "rrplugins/core/tel_api.h"
#include "telplugins_types.h"
#include "Poco/ClassLibrary.h"
//---------------------------------------------------------------------------


namespace testModel {

    using namespace tlp;

    class TestModel : public CPPPlugin
    {
    public:
        TestModel();
        ~TestModel();
        bool                        execute(bool);
        void                        resetPropertiesValues();
        unsigned char*              getManualAsPDF() const;
        size_t                      getPDFManualByteSize();

    private:
        Property<string>            mModel;
        Property<TelluriumData>     mTestData;
        Property<TelluriumData>     mTestDataWithNoise;
        Property<double>            mSigma;
        Property<double>            mNumPoints;
        Property<double>            mTimeEnd;
        Property<unsigned long>     mSeed;
        void                        addWeights();
        void                        assignPropertyDescriptions();
    };

    #ifdef EXPORT_TEST_MODEL
    extern "C"
    {
        TLP_DS TestModel*  plugins_cc createPlugin();
        TLP_DS const char* plugins_cc getImplementationLanguage();
        TLP_DS void        plugins_cc setHostInterface(rrc::THostInterface* _hostInterface);
        TLP_DS void        plugins_cc setPluginManager(tlpc::TELHandle manager);
    }
#endif
}

#endif
