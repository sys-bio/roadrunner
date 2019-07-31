#pragma hdrstop
#include <sstream>
#include "rrLogger.h"
#include "rrException.h"
#include "chisquare_doc.h"
#include "csChiSquare.h"
#include "telTelluriumData.h"
#include "rrUtils.h"
//---------------------------------------------------------------------------

using namespace std;
using rr::StringList;
ChiSquare::ChiSquare()
:
CPPPlugin(                      "ChiSquare", "Misc",       NULL, NULL),

//Properties.                   //value,                name,                                   hint,                                                           description, alias, readonly);
mExperimentalData(              TelluriumData(),        "ExperimentalData",                     "Data object holding Experimental data (input)"),
mModelData(                     TelluriumData(),        "ModelData",                            "Data object holding model data (input)"),
mNrOfModelParameters(           0,                      "NrOfModelParameters",                  "Number of model parameters (input)", "", "", true),
mChiSquare(                     0,                      "ChiSquare",                            "Chi-Square (output)", "", "", true),
mReducedChiSquare(              0,                      "ReducedChiSquare",                     "Reduced Chi-Square (output)", "", "", true),
mWorker(*this)
{
    mVersion = "1.0.0";

    //Add plugin properties to property container
    mProperties.add(&mExperimentalData);
    mProperties.add(&mModelData);
    mProperties.add(&mNrOfModelParameters);

    mProperties.add(&mChiSquare);
    mProperties.add(&mReducedChiSquare);

    mHint = "Calculate Chisquare and Reduced Chisquare.";
    mDescription="The purpose of the ChiSquare plugin is to calculate the ChiSquare and the reduced ChiSquare, for two sets of data.";

    //The function below assigns property descriptions
    assignPropertyDescriptions();
}

ChiSquare::~ChiSquare()
{}

bool ChiSquare::isWorking() const
{
    return mWorker.isRunning();
}

unsigned char* ChiSquare::getManualAsPDF() const
{
    return pdf_doc;
}

unsigned int ChiSquare::getPDFManualByteSize()
{
    return sizeofPDF;
}

string ChiSquare::getImplementationLanguage()
{
    return ::getImplementationLanguage();
}

bool ChiSquare::resetPlugin()
{
    if(mWorker.isRunning())
    {
        return false;
    }

    mTerminate = false;
    return true;
}

bool ChiSquare::execute(bool inThread)
{
    try
    {
        Log(lInfo)<<"Executing the ChiSquare plugin";
        mWorker.start(inThread);
        return true;
    }
    catch(const rr::Exception& ex)
    {
        Log(lError) << "There was a problem in the execute function of the ChiSquare plugin: " << ex.getMessage();
        throw(ex);
    }
    catch(...)
    {
        Log(lError) << "There was an unknown problem in the execute of the ChiSquare plugin.";
        throw("There was an unknown problem in the execute of the ChiSquare plugin.");
    }
}

// Plugin factory function
ChiSquare* plugins_cc createPlugin()
{
    //allocate a new object and return it
    return new ChiSquare();
}

const char* plugins_cc getImplementationLanguage()
{
    return "CPP";
}

void ChiSquare::assignPropertyDescriptions()
{
    stringstream s;

s << "Experimental data is used for input.";
mExperimentalData.setDescription(s.str());
s.str("");

s << "Model data is used for input.";
mModelData.setDescription(s.str());
s.str("");

s << "Number of model parameters is used for input.";
mNrOfModelParameters.setDescription(s.str());
s.str("");

s << "The Chisquare is the output.";
mChiSquare.setDescription(s.str());
s.str("");

s << "The Reduced Chisquare is part of the output.";
mReducedChiSquare.setDescription(s.str());
s.str("");

}

