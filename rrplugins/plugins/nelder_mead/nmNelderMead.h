#ifndef nmNelderMeadH
#define nmNelderMeadH
#include <vector>
#include "rr-libstruct/lsMatrix.h"
#include "lib/nmsimplex.h"
#include "telProperty.h"
#include "telCPPPlugin.h"
#include "nmWorker.h"
#include "telCPPPlugin.h"
//---------------------------------------------------------------------------

namespace nmfit
{
    using rr::RoadRunner;
    using std::string;
    using namespace tlp;

    class NelderMead : public CPPPlugin
    {
        friend class nmWorker;

    public:
        Property<string>                        mSBML;                          //This is the model
        Property<TelluriumData>				    mExperimentalData;
        Property<TelluriumData>			        mModelData;

        Property<Properties>                    mInputParameterList;            //Parameters to fit
        Property<Properties>                    mOutputParameterList;           //Parameters that was fitted
        Property<Properties>                    mConfidenceLimits;              //Confidence limits for each parameter

        Property<tlp::StringList>               mExperimentalDataSelectionList; //Species selection list for observed data
        Property<tlp::StringList>               mModelDataSelectionList;        //Species selection list for observed data
        Property<int>                           mNrOfIter;                      //Part of minimization result
        Property<int>                           mNrOfFuncIter;                  //Part of minimization result

        //Nelder-Mead tuning parameters
        Property<double>                        mEpsilon;
        Property<double>                        mScale;
        Property<int>                           mMaxIterations;                 /* maximum number of iterations */
        Property<double>                        mALPHA;                         /* reflection coefficient */
        Property<double>                        mBETA;                          /* contraction coefficient */
        Property<double>                        mGAMMA;                         /* expansion coefficient */

        //Output data
        Property<string>                        mStatusMessage;                 //Message regarding the status of the fit
        Property<double>                        mNorm;                          //Part of minimization result
        Property<TelluriumData>                 mNorms;                         //Norm values from the fitting
        TelluriumData&                          rNormsData;                     //Setup a reference to Norms Data

        Property<TelluriumData>			        mResidualsData;                 //Residuals from the fitting
        Property<TelluriumData>			        mStandardizedResiduals;         //Standardized Residuals from the fitting
        Property<TelluriumData>			        mNormalProbabilityOfResiduals;  //Normal probability of residuals, Q-Q plot
        Property<double>			            mChiSquare;                     //Chi square for the fitting
        Property<double>			            mReducedChiSquare;              //Reduced Chi Square

        Property<TelluriumData>                 mHessian;                       //Hessian
        Property<TelluriumData>                 mCovarianceMatrix;              //Covariance Matrix

        vector<double>                          mTheNorms;                      //For effiency
       //Utility functions for the thread
        string                                  getSBML();

    protected:
        //The worker is doing the work
        nmWorker                                mWorker;
        Plugin* mChiSquarePlugin;

    public:
        NelderMead();
        ~NelderMead();

        bool                                    execute(bool inThread = false);
        string                                  getResult();
        bool                                    resetPlugin();
        string                                  getImplementationLanguage();
        string                                  getStatus();
        bool                                    isWorking() const;

        unsigned char* getManualAsPDF() const;
        size_t                                  getPDFManualByteSize();
        tlp::StringList                         getExperimentalDataSelectionList();
        void                                    assignPropertyDescriptions();
        Plugin* getChiSquarePlugin();
    };

    extern "C"
    {
        TLP_DS NelderMead* plugins_cc  createPlugin();
        TLP_DS const char* plugins_cc  getImplementationLanguage();
        TLP_DS void        plugins_cc  setHostInterface(rrc::THostInterface* _hostInterface);
        TLP_DS void        plugins_cc  setPluginManager(tlpc::TELHandle manager);
    }

}
#endif
