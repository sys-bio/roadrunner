#ifndef add_noiseH
#define add_noiseH
#include "telProperty.h"
#include "telCPPPlugin.h"
#include "telTelluriumData.h"
#include "add_noise_worker.h"

//---------------------------------------------------------------------------
namespace addNoise
{

using namespace rr;

class AddNoise : public CPPPlugin
{
    public:
        friend class AddNoiseWorker;
        enum NoiseType {ntGaussian = 0, ntPsychological, ntUndefined};

    private:
        Property<int>               mNoiseType;
        Property<double>            mSigma;
        Property<TelluriumData>     mData;
        Property<double>            mProgress;
        AddNoiseWorker              mAddNoiseWorker;
        void                        assignPropertyDescriptions();

    public:
                                    AddNoise(RoadRunner* aRR = NULL, PluginEvent fn1 = NULL, PluginEvent fn2 = NULL, PluginEvent fn3 = NULL);
                                   ~AddNoise();

                                
        bool                        execute(bool inThread = false);
        bool                        isWorking() const; //Returns true as long the thread is active..
        unsigned char*              getManualAsPDF() const;
        unsigned int                getPDFManualByteSize();
};

extern "C"
{
RR_DS Plugin*      plugins_cc createPlugin();
RR_DS const char*  plugins_cc getImplementationLanguage();
}

}

namespace rr
{

template<>
inline string getPropertyType(const addNoise::AddNoise::NoiseType& value)
{
    return "NoiseType";
}

template<>
inline string Property<addNoise::AddNoise::NoiseType>::getValueAsString() const
{
    switch(mValue)
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
    if(val == "0")
    {
        mValue = addNoise::AddNoise::ntGaussian;
    }
    else if(val == "1")
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
