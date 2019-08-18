#pragma hdrstop
#include "deDifferentialEvolution.h"
#include <sstream>
#include "rrRoadRunner.h"
#include "rrException.h"
#include "rrLogger.h"
#include "rrException.h"
#include "telTelluriumData.h"
#include "rrUtils.h"
#include "telPluginManager.h"
//---------------------------------------------------------------------------

using namespace std;
using rr::StringList;

DifferentialEvolution::DifferentialEvolution(PluginManager* manager)
:
CPPPlugin(                      "Differential-Evolution",          "Fitting",       NULL, manager),

//Properties.                   //value,                name,                                   hint,                                                           description, alias, readonly);
mSBML(                          "<none>",               "SBML",                                 "SBML document as a string. Model to be used in the fitting"),
mExperimentalData(              TelluriumData(),        "ExperimentalData",                     "Data object holding Experimental data: Provided by client"),
mModelData(                     TelluriumData(),        "FittedData",                           "Data object holding model data: Handed to client"),
mResidualsData(                 TelluriumData(),        "Residuals",                            "Data object holding residuals: Handed to client", "", "", true),
mInputParameterList(            Properties(),           "InputParameterList",                   "List of parameters to fit"),
mOutputParameterList(           Properties(),           "OutputParameterList",                  "List of parameters that was fitted"),
mConfidenceLimits(              Properties(),           "ConfidenceLimits",                     "Confidence limits for each parameter"),
mExperimentalDataSelectionList( StringList(),           "ExperimentalDataSelectionList",        "Experimental data selection list"),
mModelDataSelectionList(        StringList(),           "FittedDataSelectionList",              "Fitted data selection list"),
mNorm(                          0,                      "Norm",                                 "Norm of fitting. An estimate of goodness of fit"),
mNorms(                         TelluriumData(),        "Norms",                                "Norms from fitting session.", "", "", true),
mNrOfIter(                      0,                      "NrOfIter",                             "Number of iterations"),
mNrOfFuncIter(                  0,                      "NrOfFuncIter",                         "Number of objective function iterations"),
mHessian(                       ls::Matrix<double>(),   "Hessian",                              "Hessian Matrix"),
mCovarianceMatrix(              ls::Matrix<double>(),   "CovarianceMatrix",                     "Covariance Matrix"),

mStandardizedResiduals(         TelluriumData(),        "StandardizedResiduals",                "Standarized residuals.", "", "", true),
mNormalProbabilityOfResiduals(  TelluriumData(),        "NormalProbabilityOfResiduals",         "Normal Probability of Residuals.", "", "", true),
mChiSquare(                     0,                      "ChiSquare",                            "Chi-Square after fitting", "", "", true),
mReducedChiSquare(              0,                      "ReducedChiSquare",                     "Reduced Chi-Square after fitting", "", "", true),
mStatusMessage(                 "<none>",               "StatusMessage",                        "Status message from fitting engine", "", "", true),

//The following Properties are properties controlling the internal working of the algorithm.
//Changing their default values may be needed depending on the problem.
mCR(                        0.5,                  "CrossOverProbability",                               "CrossOverProbability."),
mF(                         0.7,                      "Differential Weight",                                 "Differential Weight."),
mMaxIterations(             10000,                   "MaxNrOfIterations",                     "Maximum number of iterations."),
mWorker(*this),
mRRI(NULL),
rNormsData(mNorms.getValueReference())
{
    mVersion = "1.0.0";

    //Add plugin properties to property container
    mProperties.add(&mSBML);
    mProperties.add(&mExperimentalData);
    mProperties.add(&mModelData);
    mProperties.add(&mResidualsData);
    mProperties.add(&mInputParameterList);
    mProperties.add(&mOutputParameterList);
    mProperties.add(&mConfidenceLimits);
    mProperties.add(&mExperimentalDataSelectionList);
    mProperties.add(&mModelDataSelectionList);
    mProperties.add(&mNorm);
    mProperties.add(&mNorms);
    mProperties.add(&mNrOfIter);
    mProperties.add(&mNrOfFuncIter);
    mProperties.add(&mStandardizedResiduals);
    mProperties.add(&mNormalProbabilityOfResiduals);
    mProperties.add(&mChiSquare);
    mProperties.add(&mReducedChiSquare);
    mProperties.add(&mHessian);
    mProperties.add(&mCovarianceMatrix);

    //Add the DE parameters
    mProperties.add(&mF);
    mProperties.add(&mCR);
    mProperties.add(&mMaxIterations);

    mProperties.add(&mStatusMessage);

    //Allocate model and Residuals data
    mResidualsData.setValue(new TelluriumData());
    mModelData.setValue(new TelluriumData());

    mHint       ="Parameter fitting using the Differential-Evolution algorithm";
    mDescription="The Differential-Evolution plugin is used to fit a proposed \
SBML models parameters to experimental data. \
The current implementation is based on the Differential-Evolution https://en.wikipedia.org/wiki/Differential_evolution. \
The Plugin has a number of parameters for fine tuning the algorithm.";
    //The function below assigns property descriptions
    assignPropertyDescriptions();
}


DifferentialEvolution::~DifferentialEvolution()
{}

RoadRunner* DifferentialEvolution::getRoadRunner()
{
    return mRRI;
}

Plugin* DifferentialEvolution::getChiSquarePlugin()
{
    return mChiSquarePlugin;
}

bool DifferentialEvolution::isWorking() const
{
    return mWorker.isRunning();
}

StringList DifferentialEvolution::getExperimentalDataSelectionList()
{
    return mExperimentalDataSelectionList.getValue();
}

string DifferentialEvolution::getStatus()
{
    stringstream msg;
    msg<<Plugin::getStatus();
    msg<<"\nFitting parameters: "<<mInputParameterList;
    msg <<getResult();
    return msg.str();
}

string DifferentialEvolution::getImplementationLanguage()
{
    return ::getImplementationLanguage();
}

bool DifferentialEvolution::resetPlugin()
{
    if(mWorker.isRunning())
    {
        return false;
    }

    mTerminate = false;
    mInputParameterList.getValueReference().clear();
    mOutputParameterList.getValueReference().clear();
    mExperimentalDataSelectionList.getValueReference().clear();
    mModelDataSelectionList.getValueReference().clear();

    //Clear data
    mExperimentalData.clearValue();
    mModelData.clearValue();
    mNrOfIter.clearValue();
    mNorms.clearValue();
    mResidualsData.clearValue();
    return true;
}

string DifferentialEvolution::getSBML()
{
    return mSBML.getValue();
}

string DifferentialEvolution::getResult()
{
    stringstream msg;
    Properties& pars = mOutputParameterList.getValueReference();
    Properties& conf = mConfidenceLimits.getValueReference();

    for(int i = 0; i < pars.count(); i++)
    {
        Property<double>* prop = dynamic_cast< Property<double>* > (pars[i]);
        Property<double>* confProp = dynamic_cast<Property<double>* > (conf[i]);
        msg<<prop->getName()<<" = "<< prop->getValue() <<" +/- "<<confProp->getValue()<<"\n";
    }

    msg<<"Norm: "<<mNorm.getValue()<<endl;
    msg<<"Chi Square: "<<mChiSquare.getValue()<<endl;
    msg<<"Reduced Chi Square: "<<mReducedChiSquare.getValue()<<endl;
//    msg<<"Fit Engine Status: "<<mStatusMessage.getValue()<<endl;
    return msg.str();
}

bool DifferentialEvolution::execute(bool inThread)
{
    const PluginManager *PM = this->getPluginManager();
    mChiSquarePlugin = PM->getPlugin("ChiSquare");
    if(!mChiSquarePlugin)
    {
        throw(Exception("ChiSquare plugin is NULL!"));
    }

    stringstream msg;
    try
    {
        Log(lInfo)<<"Executing the Differential-Evolution plugin";
        mWorker.start(inThread);
        return true;
    }
    catch(const Exception& ex)
    {
        msg << "There was a problem in the execute of the Differential-Evolution plugin: " << ex.getMessage();
        throw(Exception(msg.str()));
    }
    catch(...)
    {
        throw(Exception("There was an unknown problem in the execute method of the Differential-Evolution plugin."));
    }
}

// Plugin factory function
DifferentialEvolution* plugins_cc createPlugin(void* manager)
{
    //allocate a new object and return it
    return new DifferentialEvolution((PluginManager*) manager);
}

const char* plugins_cc getImplementationLanguage()
{
    return "CPP";
}

void DifferentialEvolution::assignPropertyDescriptions()
{
    stringstream s;
s << "The SBML property should be assigned the (XML) \
text that defines the SBML model that is used to fit parameters.";
mSBML.setDescription(s.str());
s.str("");

s << "Experimental data contains the data to be used for fitting input.";
mExperimentalData.setDescription(s.str());
s.str("");

s << "Model data is calculated after the fitting algorithm finishes. It uses the obtained model parameters as input.";
mModelData.setDescription(s.str());
s.str("");

s << "Residuals data contains the differencies between the Experimental data and the ModelData.";
mResidualsData.setDescription(s.str());
s.str("");

s << "The input parameter list holds the parameters, and their initial values that are to be fitted, e.g. k1, k2. \
The input parameters are properties of the input SBML model";
mInputParameterList.setDescription(s.str());
s.str("");

s << "The output parameter list holds the resulting fitted parameter(s)";
mOutputParameterList.setDescription(s.str());
s.str("");

s << "The confidence limits parameter list holds resulting confidence limits, as calculated from the Hessian";
mConfidenceLimits.setDescription(s.str());
s.str("");

s << "The data input may contain multiple columns of data. The Experimental data selection list \
should contain the columns in the input data that is intended to be used in the fitting.";
mExperimentalDataSelectionList.setDescription(s.str());
s.str("");

s << "The model data selection list contains the selections for which model data will be genereated.  \
Model data can only be generated for selections present in the experimental data selectionlist.";
mModelDataSelectionList.setDescription(s.str());
s.str("");

s << "The norm is a readonly output variable indicating the goodness of fit. The smaller value, the better fit.";
mNorm.setDescription(s.str());
s.str("");

s << "The norm is calculated throughout a fitting session. Each Norm value is stored in the Norms (readonly) variable.";
mNorms.setDescription(s.str());
s.str("");

s << "The number of iterations will hold the number of iterations of the internal fitting routine.";
mNrOfIter.setDescription(s.str());
s.str("");

s << "The number of function iterations (NrOfFuncIter) wil hold the number of times the objective function was called.";
mNrOfFuncIter.setDescription(s.str());
s.str("");

s << "The status message give information on the status of an obtained fit (currently unused).";
mStatusMessage.setDescription(s.str());
s.str("");

s << "Hessian matrix. The Hessian is calculated using approximation at a found parameter minimum.";
mHessian.setDescription(s.str());
s.str("");

s << "Covariance matrix. Calculated as the inverse of the Hessian.";
mCovarianceMatrix.setDescription(s.str());
s.str("");

s << "Covariance matrix. Calculated as the inverse of the Hessian.";
mCovarianceMatrix.setDescription(s.str());
s.str("");

s << "Standardized residuals are the residuals normalized to a normal distribution.";
mStandardizedResiduals.setDescription(s.str());
s.str("");

s << "Normal Probability of Residuals, i.e. Q-Q data.";
mNormalProbabilityOfResiduals.setDescription(s.str());
s.str("");

s << "The calculated ChiSquare at the found minimum.";
mChiSquare.setDescription(s.str());
s.str("");

s << "The calculated reduced ChiSquare at the found minimum.";
mReducedChiSquare.setDescription(s.str());
s.str("");

//Add internal DifferentialEvolution properties
s << "Maximum number of iterations.";
mMaxIterations.setDescription(s.str());
s.str("");

s << "Cross Over Probability.";
mCR.setDescription(s.str());
s.str("");

s << "Differential Difference";
mF.setDescription(s.str());
s.str("");

}
