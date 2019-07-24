#ifndef psParameterScanH
#define psParameterScanH
#include <vector>
#include "telProperty.h"
#include "telCPPPlugin.h"
#include "psParameterScanWorker.h"
#include "telArrayedParameter.h"
//---------------------------------------------------------------------------

using namespace tlp;
using std::string;
using std::vector;

class ParameterScan : public CPPPlugin
{
    friend class ParameterScanWorker;

    public:
        //Input
        Property<string>                        mSBML;
        Property<string>                        mSelectionList;
        Property< ArrayedParameter >            mParameter; //Parameter to scan

        Property<double>                        mStartTime;
        Property<double>                        mEndTime;
        Property<int>                           mNumberOfPointsPerSimulation;

        //Output
        Property<TelluriumData>				    mOutputData;

    protected:
        ParameterScanWorker                     mWorker;

    public:
                                                ParameterScan();
                                               ~ParameterScan();
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
TLP_DS ParameterScan*   plugins_cc       createPlugin();
TLP_DS const char*      plugins_cc       getImplementationLanguage();
}

#endif
