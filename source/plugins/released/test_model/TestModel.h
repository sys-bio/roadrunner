#ifndef TestModelH
#define TestModelH
#include "telProperty.h"
#include "telCPPPlugin.h"
#include "telTelluriumData.h"
//---------------------------------------------------------------------------
using namespace rr;

class TestModel : public CPPPlugin
{
    public:
                                    TestModel(PluginManager* manager);
                                   ~TestModel();
        bool                        execute(bool);
        unsigned char*              getManualAsPDF() const;
        unsigned int                getPDFManualByteSize();

    private:
        Property<string>            mModel;
        Property<TelluriumData>     mTestData;
        Property<TelluriumData>     mTestDataWithNoise;
        Property<double>            mSigma;
        void                        addWeights();
        void                        assignPropertyDescriptions();
};

extern "C"
{
RR_DS Plugin*      plugins_cc createPlugin(void* manager);
RR_DS const char*  plugins_cc getImplementationLanguage();
}
#endif
