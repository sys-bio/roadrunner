#ifndef telAutoTelluriumInterfaceH
#define telAutoTelluriumInterfaceH
#include "rrRoadRunner.h"
#include "telAutoConstants.h"
#include "tai_exporter.h"
#include "telProperty.h"
#include "rrLogger.h"
#include "rrStringList.h"
#include "telProperties.h"
//---------------------------------------------------------------------------

#if defined(_MSC_VER) || defined(__BORLANDC__)
    #define autoCallConv __cdecl
#else
    #define autoCallConv
#endif

//This class serves as a RoadRunner-Auto interface. It needs a roadrunner instance on creation
namespace telauto
{

using std::string;
using rr::RoadRunner;
using rr::StringList;
using rr::Properties;

class TA_DS AutoTellurimInterface
{
    public:
                                    AutoTellurimInterface(RoadRunner* rr);
                                   ~AutoTellurimInterface();
        string                      getTempFolder();
        bool                        setTempFolder(const string& fldr);
        bool                        setupUsingCurrentModel();
        string                      getConstantsAsString();
        void                        run();
        bool                        selectParameter(const string& para);
        bool                        setScanDirection(ScanDirection val);
		//bool						setPreSimulation(bool val);
        void                        assignRoadRunner(RoadRunner* rrInstance);
        void                        assignProperties(Properties* props);

    private:
        static string               mPCPParameterName;
        static StringList           mModelParameters;
        static StringList           mModelBoundarySpecies;
        string                      mTempFolder;

        static RoadRunner*          mRR;            //Static so we can access this in autos callback
        static Properties*          mProperties;
        static AutoConstants        mAutoConstants;

        void                        setInitialPCPValue();
        static int  autoCallConv    ModelInitializationCallback(long ndim, double t, double* u, double* par);
        static void autoCallConv    ModelFunctionCallback(const double* oVariables, const double* par, double* oResult);
};
}

namespace rr
{

template<>
inline string getPropertyType<telauto::ScanDirection>(const telauto::ScanDirection& a)
{
    return "ScanDirection";
}

}

#endif
