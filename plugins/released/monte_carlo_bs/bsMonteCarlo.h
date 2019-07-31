#ifndef bsMonteCarloH
#define bsMonteCarloH
#include <vector>
#include "telProperty.h"
#include "telCPPPlugin.h"
#include "telTelluriumData.h"
#include "bsWorker.h"

//---------------------------------------------------------------------------
using namespace rr;
using std::string;

class MonteCarlo : public CPPPlugin
{
    friend class bsWorker;

    protected:
        Property<string>                        mSBML;                          //This is the model
        Property<TelluriumData>				    mExperimentalData;

        Property<Properties>                    mInputParameterList;            //Parameters to fit

        Property<Properties>                    mCurrentParameters;            //This property can be checked in OnProgress event
        Property<TelluriumData>                 mMonteCarloParameters;         //Obtained parameter values from each MC run

        Property<Properties>                    mConfidenceLimits;              //Final Confidence limits for each parameter

        Property<StringList>                    mExperimentalDataSelectionList;
        Property<StringList>                    mModelDataSelectionList;
        Property<int>                           mNrOfMCRuns;
        Property<string>                        mMinimizerPlugin;

		//Utility functions for the thread
        string                                  getTempFolder();
        string                                  getSBML();

    protected:

        //The worker is doing the work
        bsWorker                                mWorker;

    public:
                                                MonteCarlo(PluginManager* manager);
                                               ~MonteCarlo();

        bool                                    execute(bool inThread = false);
        string                                  getResult();
        bool                                    resetPlugin();
        string                                  getImplementationLanguage();
        string                                  getStatus();
        bool                                    isWorking() const;

        unsigned char*                          getManualAsPDF() const;
        unsigned int                            getPDFManualByteSize();
        rr::StringList                         getExperimentalDataSelectionList();
        void                                    assignPropertyDescriptions();
};

extern "C"
{
RR_DS MonteCarlo*    plugins_cc    createPlugin(void* manager);
RR_DS const char*    plugins_cc    getImplementationLanguage();
}




#endif
