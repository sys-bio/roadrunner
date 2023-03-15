#ifndef add_noiseH
#define add_noiseH
#include "telProperty.h"
#include "rrplugins/pluginBaseClass/telCPPPlugin.h"
#include "telTelluriumData.h"
#include "add_noise_worker.h"
#include "rrplugins/core/tel_api.h"
#include "Poco/ClassLibrary.h"

//---------------------------------------------------------------------------
namespace addNoise
{
    using namespace tlp;

    class AddNoise : public CPPPlugin
    {
    public:
        friend class AddNoiseWorker;
        enum NoiseType { ntGaussian = 0, ntPsychological, ntUndefined };

    private:
        Property<int>               mNoiseType;
        Property<double>            mSigma;
        Property<TelluriumData>     mData;
        Property<double>            mProgress;
        Property<unsigned long>     mSeed;
        Property<bool>              mAssumeTime;
        AddNoiseWorker              mAddNoiseWorker;
        void                        assignPropertyDescriptions();

    public:
        AddNoise(rrc::THostInterface* aRR = NULL, PluginEvent fn1 = NULL, PluginEvent fn2 = NULL, PluginEvent fn3 = NULL);
        ~AddNoise();


        bool                        execute(bool inThread = false);
        bool                        isWorking() const; //Returns true as long the thread is active..
        unsigned char* getManualAsPDF() const;
        size_t                      getPDFManualByteSize();
    };

#ifdef EXPORT_ADD_NOISE

    extern "C"
    {
        TLP_DS Plugin* plugins_cc createPlugin();
        TLP_DS const char* plugins_cc getImplementationLanguage();
        TLP_DS void         plugins_cc setHostInterface(rrc::THostInterface* _hostInterface);
    }

#endif

}

namespace tlp
{

    template<>
    inline string getPropertyType(const addNoise::AddNoise::NoiseType& value)
    {
        return "NoiseType";
    }

    template<>
    inline string Property<addNoise::AddNoise::NoiseType>::getValueAsString() const
    {
        switch (mValue)
        {
        case addNoise::AddNoise::ntGaussian:
            return "Gaussian";
        case addNoise::AddNoise::ntPsychological:
            return "SomethingElse";
        case addNoise::AddNoise::ntUndefined:
            return "Undefined";
        }
        return "";
    }

    template<>
    inline void Property< addNoise::AddNoise::NoiseType >::setValueFromString(const string& val)
    {
        //Only gaussian noise is available at this time
        if (val == "0")
        {
            mValue = addNoise::AddNoise::ntGaussian;
        }
        else if (val == "1")
        {
            mValue = addNoise::AddNoise::ntPsychological;
        }
        else
        {
            mValue = addNoise::AddNoise::ntUndefined;
        }
    }

}

#endif
