#ifndef telAutoPluginH
#define telAutoPluginH
#include <vector>
#include "telProperty.h"
#include "telCPPPlugin.h"
#include "telPluginManager.h"
#include "telAutoWorker.h"
#include "libAutoTelluriumInterface/telAutoConstants.h"
#include "libAutoTelluriumInterface/telAutoTelluriumInterface.h"
//---------------------------------------------------------------------------

using telauto::AutoTellurimInterface;
using rr::Property;
using std::vector;
class AutoPlugin : public rr::CPPPlugin
{
    friend class AutoWorker;
    public:
                                                AutoPlugin();
                                               ~AutoPlugin();
        //Data input
        void                                    setScanDirection(telauto::ScanDirection dir);
        bool                                    execute(bool inThread = false);
        string                                  getResult();
        string                                  getConstants();
        bool                                    resetPlugin();
        bool                                    setInputData(void* data);
        bool                                    isWorking() const;
        telauto::AutoTellurimInterface&         getRRAuto();
        unsigned char*                          getManualAsPDF() const;
        unsigned int                            getPDFManualByteSize();

    protected:

        //The interface to auto. Takes the property container as a reference
        AutoTellurimInterface                   mRRAuto;

        //The worker
        AutoWorker                              mAutoWorker;

        Property<string>                        mSBML;                  //This is the model
        Property<string>                        mTempFolder;
        Property<bool>                          mKeepTempFiles;
        Property<string>                        mScanDirection;         //How auto sweeps the parameter
		Property<bool>							mPreSimulation;
		Property<double>						mPreSimulationDuration;
		Property<double>						mPreSimulationStart;
		Property<int>							mPreSimulationSteps;

        Property<string>                        mPrincipalContinuationParameter;

        Property< TelluriumData >               mBifurcationData;
        Property< vector<int> >                 mBifurcationPoints;
        Property< StringList >                  mBifurcationLabels;

        Property<bool>                          mCaptureOutputFiles;
        Property<bool>                          mAllowConservedMoiety;
        Property<string>                        mFort2;
        Property<string>                        mFort3;
        Property<string>                        mFort6;     //Alias BifurcationSummary
        Property<string>                        mFort7;     //Alias BifurcationDiagram
        Property<string>                        mFort8;
        Property<string>                        mFort9;

        //Auto parameters
        // NDIM: dimension of the system of equations, as specified in the user-supplied subroutine 'func'
        Property<int>                           mNDIM;

        /// IPS: Constant defining the problem type (-2 -> 14) : 1 : stationary states 2 : periodic solutions 4 : BVP
        Property<int>                           mIPS;

        /// Restart label
        Property<int>                           mIRS;

        /// Locate limit points (folds)
        Property<int>                           mILP;

        /// the number of free parameters
        Property<int>                           mNICP;

        /// The free parameters
        Property< vector<int> >                 mICP;

        /// the number of mesh intervals
        Property<int>                           mNTST;

        /// the number of collocation points per mesh interval
        Property<int>                           mNCOL;

        /// Mesh adaption:
        ///     0 :fixedmesh
        ///   > 0 : adapt mesh every IAD continuation steps
        Property<int>                           mIAD;

        /// Detection of branch points:
        ///   0 : don’t detect
        ///   1 : detect for algebraic problems
        ///   2 : detect for all problems
        Property<int>                           mISP;

        /// At a restart solution:
        ///   1 : just follow the branch
        ///  -1 : switch branches (for BPs in BVPs)
        ///   2 : two-parameter continuation (LPs, HBs, PDs, TRs)
        Property<int>                           mISW;

        /// The ”principal solution measure”
        /// (the second real column in the bifurcation diagram file):
        ///      0 : the L2 norm
        ///   != 0 : other measures (see manual)
        Property<int>                           mIPLT;

        /// the number of boundary conditions (for BVP)
        Property<int>                           mNBC;

        /// the number of integral constraints (for BVP)
        Property<int>                           mNINT;

        /// maximum number of continuation step along any branch
        Property<int>                           mNMX;

        /// bifurcation diagram lower limit for the “first” parameter
        Property<double>                        mRL0;

        /// bifurcation diagram upper limit for the “first” parameter
        Property<double>                        mRL1;

        /// bifurcation diagram lower limit for the “principal solution measure”
        Property<double>                        mA0;

        /// bifurcation diagram upper limit for the “principal solution measure”
        Property<double>                        mA1;

        /// Save the solution in the solution file every NPR continuation steps
        Property<int>                           mNPR;

        /// For algebraic problems: Maximum number of branch switches (if negative only one “leg” will be computed)
        Property<int>                           mMXBF;

        /// Controls the diagnostic file output (normal value is 2; see manual)
        Property<int>                           mIID;

        /// maximum number of iterations for detecting “special” solutions
        Property<int>                           mITMX;

        /// maximum number of Newton+Chord iterations
        Property<int>                           mITNW;

        /// maximum number of Newton iterations
        Property<int>                           mNWTN;

        /// Jacobian supplied ?
        Property<int>                           mJAC;

        /// Newton/Chord convergence tolerance for free parameters
        Property<double>                        mEPSL;

        /// Newton/Chord convergence tolerance for solutions
        Property<double>                        mEPSU;

        /// Continuation step size tolerance when locating special solutions
        Property<double>                        mEPSS;

        /// Nominal continuation step size
        Property<double>                        mDS;

        /// Minimum continuation step size
        Property<double>                        mDSMIN;

        /// Maximum continuation step size
        Property<double>                        mDSMAX;

        /// Adapt the continuation step size every IADS steps
        Property<int>                           mIADS;

        /// the number of modified parameter “weights” (for BVP)
        Property<int>                           mNTHL;

        /// parameter index, parameter weight (e.g., ICP(11)=0 means PAR(11) is excluded from the step size)
        Property< vector<int> >                 mTHL;

        /// the number of modified solution component “weights” (for BVP)
        Property<int>                           mNTHU;

        /// component index, component weight
        Property< vector<int> >                 mTHU;

        /// the number of “user output points” specified
        Property<int>                           mNUZR;

        /// parameter index, parameter value (if I is negative the continuation stops at the parameter value)
        Property< vector<int> >                 mUZR;

        string                                  getTempFolder();
        string                                  getSBML();

        void                                    addProperties();
        void                                    assignPropertyDescriptions();
};

extern "C"
{
RR_DS AutoPlugin* createPlugin();
RR_DS const char* getImplementationLanguage();
}

namespace rr
{

template<>
inline string Property< telauto::ScanDirection >::getValueAsString() const
{
    return mValue == telauto::sdPositive ? "Positive" : "Negative";
}

template<>
inline void Property< telauto::ScanDirection >::setValueFromString(const string& val)
{
    mValue = compareNoCase(val, "Positive") == 0 ? telauto::sdPositive : telauto::sdNegative;
}

template<>
inline void Property< telauto::ScanDirection >::setValue(const telauto::ScanDirection& val)
{
    mValue = (val);
}
}

#endif
