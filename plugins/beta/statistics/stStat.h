#ifndef stStatH
#define stStatH
#include <vector>
#include "telProperty.h"
#include "telCPPPlugin.h"
#include "stStatWorker.h"
//---------------------------------------------------------------------------

using namespace rr;
using std::string;

class Stat : public CPPPlugin
{
    friend class stStatWorker;

    public:
        Property<TelluriumData>				    mExperimentalData;
        Property<TelluriumData>			        mModelData;
        Property<double>			            mChiSquare;                     //Chi square
        Property<double>			            mReducedChiSquare;              //Reduced Chi Square

        Property<TelluriumData>			        mResidualsData;                 //Residuals from the fitting
        Property<TelluriumData>			        mStandardizedResiduals;         //Standardized Residuals from the fitting
        Property<TelluriumData>			        mNormalProbabilityOfResiduals;  //Normal probability of residuals, Q-Q plot

    protected:
        stStatWorker                            mWorker;
    public:
                                                Stat();
                                               ~Stat();

        bool                                    execute(bool inThread = false);

        bool                                    resetPlugin();
        string                                  getImplementationLanguage();
        string                                  getStatus();
        bool                                    isWorking() const;

        unsigned char*                          getManualAsPDF() const;
        unsigned int                            getPDFManualByteSize();
        void                                    assignPropertyDescriptions();
};

extern "C"
{
RR_DS Stat*        plugins_cc       createPlugin();
RR_DS const char*  plugins_cc       getImplementationLanguage();
}

#endif
