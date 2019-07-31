#ifndef lmH
#define lmH
#include <vector>
#include "telProperty.h"
#include "telCPPPlugin.h"
#include "rr-libstruct/lsMatrix.h"
#include "lmWorker.h"
#include "lib/lmmin.h"
//---------------------------------------------------------------------------

namespace lmfit
{
using namespace rr;
using rr::RoadRunner;
using std::string;

class LM : public CPPPlugin
{
    friend class lmWorker;

    public:
        Property<string>                        mSBML;                          //This is the model
        Property<TelluriumData>				    mExperimentalData;
        Property<TelluriumData>			        mModelData;

        Property<Properties>                    mInputParameterList;            //Parameters to fit
        Property<Properties>                    mOutputParameterList;           //Parameters that was fitted
        Property<Properties>                    mConfidenceLimits;              //Confidence limits for each parameter

        Property<rr::StringList>               mExperimentalDataSelectionList; //Species selection list for observed data
        Property<rr::StringList>               mModelDataSelectionList;        //Species selection list for observed data
        Property<int>                           mNrOfIter;                      //Part of minimization result

        //LMFIT Tuning parameters
        Property<double>                        ftol;                           /* relative error desired in the sum of squares. */
        Property<double>                        xtol;                           /* relative error between last two approximations. */
        Property<double>                        gtol;                           /* orthogonality desired between fvec and its derivs. */
        Property<double>                        epsilon;                        /* step used to calculate the jacobian. */
        Property<double>                        stepbound;                      /* initial bound to steps in the outer loop. */
        Property<int>                           patience;                       /* maximum number of iterations. */

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

        Property< ls::Matrix<double> >          mHessian;                       //Hessian
        Property< ls::Matrix<double> >          mCovarianceMatrix;              //Covariance Matrix

		//Utility functions for the thread
        string                                  getTempFolder();
        string                                  getSBML();

		lmDataStructure							&mLMData;        //LevenbergMarq.. data structure

    protected:
        //The worker is doing the work
        lmWorker                                mWorker;
        lm_status_struct                        mLMStatus;      //Check afterwards.

    public:
                                                LM(PluginManager* manager);
                                               ~LM();

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
RR_DS LM*         plugins_cc       createPlugin(void* manager);
RR_DS const char* plugins_cc       getImplementationLanguage();
}

}

namespace tlp
{

//template<>
//inline string Property< ls::Matrix<double> >::getValueAsString() const
//{
//    stringstream ss;
//    ss << mValue;
//    return ss.str();
//}

}
#endif
