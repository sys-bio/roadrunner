#ifndef csChiSquareH
#define csChiSquareH
#include <vector>
#include "telProperty.h"
#include "telCPPPlugin.h"
#include "csChiWorker.h"
//---------------------------------------------------------------------------

using namespace rr;
using std::string;

class ChiSquare : public CPPPlugin
{
    friend class ChiWorker;

    public:
        //Input
        Property<int>                           mNrOfModelParameters;
        Property<TelluriumData>				    mExperimentalData;
        Property<TelluriumData>			        mModelData;

        //Output
        Property<double>		                mChiSquare;                     //Chi square
        Property<double>		                mReducedChiSquare;              //Reduced Chi Square

    protected:
        ChiWorker                               mWorker;

    public:
                                                ChiSquare();
                                               ~ChiSquare();
        bool                                    execute(bool inThread = false);
        bool                                    resetPlugin();
        string                                  getImplementationLanguage();
        bool                                    isWorking() const;
        unsigned char*                          getManualAsPDF() const;
        unsigned int                            getPDFManualByteSize();
        void                                    assignPropertyDescriptions();
};

extern "C"
{
RR_DS ChiSquare*   plugins_cc       createPlugin();
RR_DS const char*  plugins_cc       getImplementationLanguage();
}

#endif
