#ifndef deDifferentialEvolution
#define deDifferentialEvolution
#include <vector>
#include "rr-libstruct/lsMatrix.h"
#include "telProperty.h"
#include "telCPPPlugin.h"
#include "deWorker.h"
//---------------------------------------------------------------------------

using rr::RoadRunner;
using std::string;
using namespace rr;

class DifferentialEvolution : public CPPPlugin
{
	friend class deWorker;
    public:
        Property<string>                        mSBML;                          //This is the model
        Property<TelluriumData>				          mExperimentalData;
        Property<TelluriumData>			            mModelData;

        Property<Properties>                    mInputParameterList;            //Parameters to fit
        Property<Properties>                    mOutputParameterList;           //Parameters that was fitted
        Property<Properties>                    mConfidenceLimits;              //Confidence limits for each parameter

        Property<rr::StringList>               mExperimentalDataSelectionList; //Species selection list for observed data
        Property<rr::StringList>               mModelDataSelectionList;        //Species selection list for observed data
        Property<int>                           mNrOfIter;                      //Part of minimization result
        Property<int>                           mNrOfFuncIter;                  //Part of minimization result

        //deDifferentialEvolution tuning parameters

        Property<int>                           mMaxIterations;                 /* maximum number of iterations */
        Property<double>                        mCR; /*mCR lies bw [0,1] is called the crossover probability*/
        Property<double>                        mF;/*mF lies bw [0,2] is called the differential weight*/

        //Output data
        Property<string>                        mStatusMessage;                 //Message regarding the status of the fit
        Property<double>                        mNorm;                          //Part of minimization result
        Property<TelluriumData>                 mNorms;                         //Norm values from the fitting
        TelluriumData&                          rNormsData;                     //Setup a reference to Norms Data

        Property<TelluriumData>			        mResidualsData;                 //Residuals from the fitting
        Property<TelluriumData>			        mStandardizedResiduals;         //Standardized Residuals from the fitting
        Property<TelluriumData>			        mNormalProbabilityOfResiduals;  //Normal probability of residuals, Q-Q plot
        Property<double>                        mChiSquare;                     //Chi square for the fitting
        Property<double>                        mReducedChiSquare;              //Reduced Chi Square

        Property< ls::Matrix<double> >          mHessian;                       //Hessian
        Property< ls::Matrix<double> >          mCovarianceMatrix;              //Covariance Matrix

        vector<double>                          mTheNorms;              //For effiency
 		//Utility functions for the thread
        string                                  getTempFolder();
        string                                  getSBML();

    protected:
        //The worker is doing the work
        deWorker                                mWorker;                //minimize the optimization function
        rr::RoadRunner                         *mRRI;
        Plugin*                                 mChiSquarePlugin;

    public:
                                                DifferentialEvolution(PluginManager* manager);
                                               ~DifferentialEvolution();

        bool                                    execute(bool inThread = false);
        string                                  getResult();
        bool                                    resetPlugin();
        string                                  getImplementationLanguage();
        string                                  getStatus();
        bool                                    isWorking() const;

        rr::StringList                          getExperimentalDataSelectionList();
        void                                    assignPropertyDescriptions();
        RoadRunner*                             getRoadRunner();
        Plugin*                                 getChiSquarePlugin();
};

extern "C"
{
RR_DS DifferentialEvolution* plugins_cc       createPlugin(void* manager);
RR_DS const char* plugins_cc       getImplementationLanguage();
}

#endif
