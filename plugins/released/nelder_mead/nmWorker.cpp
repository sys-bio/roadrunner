#pragma hdrstop
#include "nmWorker.h"
#include "rrRoadRunnerOptions.h"
#include "rr-libstruct/lsLA.h"
#include "wrappers/C/rrc_api.h" //Todo: no reason using the roaddrunner C API here, convert and use the CPP api directly
#include "wrappers/C/rrc_utilities.h"
#include "lib/nmsimplex.h"
#include "rrLogger.h"
#include "rrException.h"
#include "telTelluriumData.h"
#include "rrUtils.h"
#include "telMathUtils.h"
#include "telProperty.h"
#include "telPluginManager.h"
#include "nmNelderMead.h"
#include "nmObjectiveFunction.h"
//---------------------------------------------------------------------------
using namespace rrc;
using namespace std;
using namespace rr;

nmWorker::nmWorker(NelderMead& host)
:
mHost(host)
{}

bool nmWorker::isRunning() const
{
    return mThread.isRunning();
}

int nmWorker::getNumberOfParameters()
{
    Properties& inParas  =  mHost.mInputParameterList.getValueReference();
    return inParas.count();
}

void nmWorker::start(bool runInThread)
{
    if(runInThread)
    {
        if(mThread.isRunning())
        {
            Log(lError)<<"Tried to start an already working thread!";
            return;
        }

        mThread.start(*this);    //Will spawn the run function below in a thread
    }
    else
    {
        run();
    }
}

void nmWorker::run()
{
    workerStarted();
    setupRoadRunner();

    StringList& species = mHost.mExperimentalDataSelectionList.getValueReference();
    Log(lInfo)<<"The following species are selected: "<<species.AsString();

    Properties& inParas  =  mHost.mInputParameterList.getValueReference();
    Properties& outParas =  mHost.mOutputParameterList.getValueReference();
    int         nrOfParameters = inParas.count();

    Log(lInfo)<<"The following parameters are to be minimized";
    for(int i = 0; i < nrOfParameters; i++)
    {
        Log(lInfo)<<inParas[i]->getName()<<" with initial value: "<<inParas[i]->getValueAsString();
    }

    mHost.mNrOfIter.setValue(0);
    mHost.mNrOfFuncIter.setValue(0);
    mHost.mNorm.setValue(0.0);

    //Parameters for the Algorithm..
    double* intialParameters = new double[nrOfParameters];

    for(int i =0; i < nrOfParameters; i++)
    {
        intialParameters[i] = * (double*) inParas[i]->getValueHandle();
    }

    //Execute minimizer
    double min = simplex2(  NelderMeadObjectiveFunction,
                            &mHost,
                            intialParameters,
                            nrOfParameters,
                            mHost.mEpsilon,
                            mHost.mScale,
                            NULL,
                            mHost.mMaxIterations,
                            mHost.mALPHA,
                            mHost.mBETA,
                            mHost.mGAMMA
                            );

    //Populate with data to report back
    for (int i = 0; i < nrOfParameters; ++i)
    {
        outParas.add(new Property<double>(intialParameters[i], inParas[i]->getName(), ""), true);
    }

    //The user may have aborted the minimzation... check here..
    if(mHost.mTerminate)
    {
        //user did set the terminate flag to true.. discard any minimization data and get out of the
        //plugin execute code..
        Log(lInfo)<<"The Nelder-Mead minimization was terminated.. aborting";
        workerFinished();
        return;
    }

    //Post fitting data calculations
    Log(lDebug)<<"==================== Fitting Result ================================";
    Log(lDebug)<<"Nr of function evaluations: "  <<  mHost.mNrOfIter.getValue();
    Log(lDebug)<<"Minimized parameter values: ";
    for (int i = 0; i < inParas.count(); ++i)
    {
        Log(lDebug)<<"\t"<<outParas[i]->getName()<<" = "<< outParas[i]->getValueAsString();
    }

    Log(lDebug)<<"Final Norm:  "<<mHost.mNorm.getValue();
    postFittingWork();
    workerFinished();
}

bool nmWorker::setupRoadRunner()
{
    if(mHost.mRRI)
    {
        delete mHost.mRRI;
    }

    mHost.mRRI = new RoadRunner;
    mHost.mRRI->load(mHost.mSBML.getValue());
    mHost.mRRI->setSelections(mHost.getExperimentalDataSelectionList());
    return true;
}

void nmWorker::postFittingWork()
{
    Properties* inParas = (Properties*) mHost.mInputParameterList.getValueHandle();
    int nrOfParameters = inParas->count();

    //Create model and residuals data
    createModelData(mHost.mModelData.getValuePointer());
    createResidualsData(mHost.mResidualsData.getValuePointer());

    //Copy Norms
    mHost.rNormsData.reSize(mHost.mTheNorms.size(), 1);
    mHost.rNormsData.setColumnNames(StringList("Norm"));
    for(int r = 0; r < mHost.mTheNorms.size(); r++)
    {
        mHost.rNormsData(r,0) = mHost.mTheNorms[r];
    }

    //Calculate standardized residuals
    TelluriumData& residuals = *(TelluriumData*) mHost.mResidualsData.getValueHandle();


    //When there is a bad fit, residuals get really large and so
    //several of the statistics below overflow

    //Populate the standardized residuals
    try
    {
        TelluriumData& stdRes   = *(TelluriumData*) mHost.mStandardizedResiduals.getValueHandle();
        stdRes                  = getStandardizedPopulations(residuals);

        //Create a probability plot for the residuals
        TelluriumData& probPlot = *(TelluriumData*) mHost.mNormalProbabilityOfResiduals.getValueHandle();
        probPlot                = getNormalProbabilityPlot(stdRes);

        calculateChiSquare();
        calculateHessian();
        calculateCovariance();
        calculateConfidenceLimits();
    }
    catch(...)
    {
        Log(lError) << "There was problems calculating fit statistics";
    }

}

void nmWorker::calculateChiSquare()
{
    Properties* inParas = (Properties*) mHost.mInputParameterList.getValueHandle();
    int nrOfParameters = inParas->count();

    //Calculate ChiSquare(s)
    TelluriumData& modelData    = *(TelluriumData*) mHost.mModelData.getValuePointer();
    TelluriumData& obsData      = *(TelluriumData*) mHost.mExperimentalData.getValuePointer();

    Plugin* chi = mHost.mPM->getPlugin("tel_chisquare");

    if(!chi)
    {
        throw(Exception("Failed to get chisquare plugin in lmfit plugin"));
    }

    Property<TelluriumData>* para =  dynamic_cast<Property<TelluriumData>*>(chi->getProperty("ExperimentalData"));
    para->setValue(obsData);

    para =  dynamic_cast<Property<TelluriumData>*>(chi->getProperty("ModelData"));
    para->setValue(modelData);

    Property<int>* intPara =  dynamic_cast< Property<int>* >(chi->getProperty("NrOfModelParameters"));
    intPara->setValue(nrOfParameters);

    //Calculate Chi square
    chi->execute();

    Property<double>* chiSquare =  dynamic_cast< Property<double>* >(chi->getProperty("ChiSquare"));
    Property<double>* rChiSquare =  dynamic_cast< Property<double>* >(chi->getProperty("ReducedChiSquare"));

    mHost.mChiSquare.setValue(chiSquare->getValue());
    mHost.mReducedChiSquare.setValue(rChiSquare->getValue());

    Log(lDebug)<<"Chi Square = "<<chiSquare->getValue();
    Log(lDebug)<<"Reduced Chi Square = "<<rChiSquare->getValue();
}

double nmWorker::getChi(const Properties& parameters)
{
    Log(lDebug)<<"Getting chisquare using parameters: "<<parameters;
    //Reset RoadRunner
    mHost.mRRI->reset();

    for(int i = 0; i < parameters.count(); i++)
    {
        Property<double> *para = (Property<double>*) (parameters[i]);
        mHost.mRRI->setValue(para->getName(), para->getValue());
    }

    TelluriumData* expData = (TelluriumData*) mHost.mExperimentalData.getValueHandle();
    rr::SimulateOptions options;
    options.start       = expData->getTimeStart();
    options.duration    = expData->getTimeEnd() - expData->getTimeStart();
    options.steps       = expData->rSize() - 1;

    mHost.mRRI->reset();

    const ls::DoubleMatrix *modelData = NULL;
    if(mHost.mRRI->simulate(&options))
    {
        modelData = mHost.mRRI->getSimulationData();
    }

    TelluriumData& obsData      = *(TelluriumData*) mHost.mExperimentalData.getValuePointer();
    Plugin* chi                 = mHost.mPM->getPlugin("tel_chisquare");

    Property<TelluriumData>* para =  dynamic_cast<Property<TelluriumData>*>(chi->getProperty("ExperimentalData"));
    para->setValue(obsData);

    para =  dynamic_cast<Property<TelluriumData>*>(chi->getProperty("ModelData"));
    TelluriumData td;
    td.setData(*modelData);
    para->setValue(td);

    Property<int>* intPara =  dynamic_cast< Property<int>* >(chi->getProperty("NrOfModelParameters"));
    intPara->setValue(getNumberOfParameters());

    //Calculate Chi square
    chi->execute();
    Property<double>* chiSquare =  dynamic_cast< Property<double>* >(chi->getProperty("ChiSquare"));
    return chiSquare->getValue();
}

void nmWorker::calculateHessian()
{
    double eta = 6.06e-6;
    int nrOfParameters = mHost.mOutputParameterList.getValueReference().count();

    Properties& ref = mHost.mOutputParameterList.getValueReference();

    Properties copy; //Gotta add a copy/assignemnt opertor to Properties..
    for(int i = 0; i < ref.count(); i++)
    {
        copy.add(ref[i]);
    }

    //We need to calculate ChiSquares for various perturbatin around a parameter value, in order to get the Hessian
    DoubleMatrix mat(nrOfParameters, nrOfParameters);

    for(int i = 0; i < nrOfParameters; i++)
    {
        for(int j = 0; j < nrOfParameters; j++)
        {
            if(i == j)//Diagonal elements
            {
                Property<double>* p = (Property<double>*) copy[i];
                double origValue = p->getValue();
                double hi = eta * fabs(origValue);

                p->setValue(origValue + hi);
                double t1 = getChi(copy);

                p->setValue(origValue - hi);
                double t2 = getChi(copy);

                p->setValue(origValue);
                double t3 = getChi(copy);
                mat(i,j) = (t1 + t2 - 2.0 * t3) / (hi*hi);
            }
            else //Off diagonal elements
            {
                Property<double>* pi = (Property<double>*) copy[i];
                double origValueI = pi->getValue();

                Property<double>* pj = (Property<double>*) copy[j];
                double origValueJ = pj->getValue();
                double hi = eta * fabs(origValueI);
                double hj = eta * fabs(origValueJ);

                //Term1
                pi->setValue(origValueI + hi);
                pj->setValue(origValueJ + hj);
                double t1 = getChi(copy);

                //Term2
                pj->setValue(origValueJ);
                double t2 = getChi(copy);

                //Term3
                pi->setValue(origValueI);
                pj->setValue(origValueJ + hj);
                double t3 = getChi(copy);

                //Term4
                pi->setValue(origValueI);
                pj->setValue(origValueJ);
                double t4 = getChi(copy);
                mat(i,j) = (t1 - t2 -t3 + t4 )/(hi*hj);

            }
            //Reset parameter values

        }
    }
    mHost.mHessian.setValue(mat);
}

void nmWorker::calculateCovariance()
{
    //Check if Hessain is invertible
    DoubleMatrix mat = mHost.mHessian.getValue();

    ls::ComplexMatrix temp(mat); //Get a complex matrix from a double one. Imag part is zero

    ls::ComplexMatrix Inv = GetInverse(temp);

    DoubleMatrix temp2(mat.RSize(), mat.CSize());
    temp2 = Inv;

    mHost.mCovarianceMatrix.setValue(temp2);
}


void nmWorker::workerStarted()
{
    mHost.mIsWorking = true;
    if(mHost.mWorkStartedEvent)
    {
        mHost.mWorkStartedEvent(mHost.mWorkStartedData1, mHost.mWorkStartedData2);
    }
}

void nmWorker::workerFinished()
{
    mHost.mIsWorking = false;//Set this flag before event so client can query plugin about termination
    if(mHost.mWorkFinishedEvent)
    {
        mHost.mWorkFinishedEvent(mHost.mWorkFinishedData1, mHost.mWorkFinishedData2);
    }
}

void nmWorker::createModelData(TelluriumData* _data)
{
    TelluriumData& data = *(_data);
    TelluriumData* expData = (TelluriumData*) mHost.mExperimentalData.getValueHandle();

    Properties& outParas =  * (Properties*) mHost.mOutputParameterList.getValueHandle();
    int nrOfParameters = outParas.count();

    //We now have the parameters
    StringList selList("time");
    selList.Append(mHost.mModelDataSelectionList.getValue());

    data.reSize(expData->rSize(), selList.Count());
    mHost.mRRI->reset();
    mHost.mRRI->setSelections(selList);

    for(int i = 0; i < nrOfParameters; i++)
    {
        mHost.mRRI->setValue(outParas[i]->getName(),  * (double*) outParas[i]->getValueHandle());
    }

    rr::SimulateOptions options;
    options.start       = expData->getTimeStart();
    options.duration    = expData->getTimeEnd() - expData->getTimeStart();
    options.steps       = expData->rSize() - 1;

    mHost.mRRI->reset();

    if(mHost.mRRI->simulate(&options))
    {
        data.setData(*mHost.mRRI->getSimulationData());
    }
}

void nmWorker::createResidualsData(TelluriumData* _data)
{
    TelluriumData& resData = *(_data);
    //We now have the parameters
    TelluriumData& obsData = (mHost.mExperimentalData.getValueReference());
    TelluriumData& modData = (mHost.mModelData.getValueReference());

    resData.reSize(modData.rSize(), modData.cSize());

    //setup coulumn names
    resData.setColumnNames(modData.getColumnNames());
    StringList& species = mHost.mExperimentalDataSelectionList.getValueReference();

    for(int sel = 0; sel < species.Count() + 1; sel++)    //selection 1 becuase of time column..
    {
        for(int i = 0; i < obsData.rSize(); i++)
        {
            if(sel == 0)
            {
                resData(i, sel) = modData(i, sel);    //Time
            }
            else
            {
                //The modData may contain data for other species than that was fitted..
                //We need to find out what coulmn correspond to what..
                string specie = obsData.getColumnName(sel);
                int colNr = modData.getColumnIndex(specie);
                if(colNr != -1)
                {
                    resData(i,sel) = obsData(i, sel) - modData(i, colNr);
                }
                else
                {
                    Log(lError)<<"Problem with column names when creating residual data!";
                }
            }
        }
    }
}

void nmWorker::calculateConfidenceLimits()
{
    Properties& parameters = mHost.mOutputParameterList.getValueReference();

    Properties& conf = mHost.mConfidenceLimits.getValueReference();
    conf.clear();

    DoubleMatrix mat = mHost.mCovarianceMatrix.getValue();
    double chiReduced = mHost.mReducedChiSquare.getValue();
    for (int i = 0; i < parameters.count(); ++i)
    {
        double delta = 1.96*sqrt(mat(i,i) * chiReduced);
        conf.add(new Property<double>(delta, parameters[i]->getName() + string("_confidence"), ""), true);
    }
}

