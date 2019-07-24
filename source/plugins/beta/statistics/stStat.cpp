#pragma hdrstop
#include <sstream>
#include "telLogger.h"
#include "telException.h"
#include "stat_doc.h"
#include "stStat.h"
#include "telTelluriumData.h"
#include "telUtils.h"
//---------------------------------------------------------------------------

using namespace std;
using tlp::StringList;
Stat::Stat()
:
CPPPlugin(                      "Stat", "Misc",       NULL, NULL),

//Properties.                   //value,                name,                                   hint,                                                           description, alias, readonly);
mExperimentalData(              TelluriumData(),        "ExperimentalData",                     "Data object holding Experimental data: Provided by client"),
mModelData(                     TelluriumData(),        "FittedData",                           "Data object holding model data: Handed to client"),
mResidualsData(                 TelluriumData(),        "Residuals",                            "Data object holding residuals: Handed to client", "", "", true),

mStandardizedResiduals(         TelluriumData(),        "StandardizedResiduals",                "Standarized residuals.", "", "", true),
mNormalProbabilityOfResiduals(  TelluriumData(),        "NormalProbabilityOfResiduals",         "Normal Probability of Residuals.", "", "", true),
mChiSquare(                     0,                      "ChiSquare",                            "Chi-Square after fitting", "", "", true),
mReducedChiSquare(              0,                      "ReducedChiSquare",                     "Reduced Chi-Square after fitting", "", "", true),
mWorker(*this)
{
    mVersion = "0.8";

    //Add plugin properties to property container

    mProperties.add(&mExperimentalData);
    mProperties.add(&mModelData);
    mProperties.add(&mResidualsData);

    mProperties.add(&mStandardizedResiduals);
    mProperties.add(&mNormalProbabilityOfResiduals);
    mProperties.add(&mChiSquare);
    mProperties.add(&mReducedChiSquare);

    //Allocate model and Residuals data
    mResidualsData.setValue(new TelluriumData());
    mModelData.setValue(new TelluriumData());

    mHint ="Create various statistics from experimental and modeled data input";
    mDescription="";
    //The function below assigns property descriptions
    assignPropertyDescriptions();
}

Stat::~Stat()
{}

bool Stat::isWorking() const
{
    return mWorker.isRunning();
}

unsigned char* Stat::getManualAsPDF() const
{
    return pdf_doc;
}

unsigned int Stat::getPDFManualByteSize()
{
    return sizeofPDF;
}

string Stat::getStatus()
{
    stringstream msg;
    return msg.str();
}

string Stat::getImplementationLanguage()
{
    return ::getImplementationLanguage();
}

bool Stat::resetPlugin()
{
    if(mWorker.isRunning())
    {
        return false;
    }

    mTerminate = false;
    return true;
}


bool Stat::execute(bool inThread)
{
    try
    {
        RRPLOG(lInfo)<<"Executing the Stat plugin";
        mWorker.start(inThread);
        return true;
    }
    catch(const Exception& ex)
    {
        RRPLOG(lError) << "There was a problem in the execute of the Stat plugin: " << ex.getMessage();
        throw(ex);
    }
    catch(...)
    {
        RRPLOG(lError) << "There was an unknown problem in the execute of the StatFIT plugin.";
        throw("There was an unknown problem in the execute of the StatFIT plugin.");
    }
}

// Plugin factory function
Stat* plugins_cc createPlugin()
{
    //allocate a new object and return it
    return new Stat();
}

const char* plugins_cc getImplementationLanguage()
{
    return "CPP";
}

void Stat::assignPropertyDescriptions()
{
    stringstream s;

s << "Experimental data contains the data to be used for fitting input.";
mExperimentalData.setDescription(s.str());
s.str("");

s << "Model data is calculated after the fitting algorithm finishes. It uses the obtained model parameters as input.";
mModelData.setDescription(s.str());
s.str("");

s << "Residuals data contains the differencies between the Experimental data and the ModelData.";
mResidualsData.setDescription(s.str());
s.str("");
}

