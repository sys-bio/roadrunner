#pragma hdrstop
#include <sstream>
#include "telLogger.h"
#include "telException.h"
#include "parameter_scan_doc.h"
#include "psParameterScan.h"
#include "telTelluriumData.h"
#include "telUtils.h"
//---------------------------------------------------------------------------

using namespace std;
using tlp::StringList;
ParameterScan::ParameterScan()
:
CPPPlugin(                      "ParameterScan", "Misc",       NULL, NULL),

//Properties.                   value,                  name,                                   hint,                                                           description, alias, readonly);
mSBML(                          "<none>",               "SBML",                                 "SBML document as a string. Model to be used in parameter scan."),
mStartTime(                     0,                      "StartTime",                            "StartTime for the simulation."),
mEndTime(                       10,                     "EndTime",                              "EndTime for the simulation."),
mNumberOfPointsPerSimulation(   51,                     "NumberOfPointsPerSimulation",          "Number Of Points Per Simulation."),
mSelectionList(                 "<none>",               "SelectionList",                        "Seection List as a string."),
mOutputData(                    TelluriumData(),        "OutputData",                           "Data object holding Experimental data (input)"),
mParameter(                     ArrayedParameter() ,    "Parameter",                            "One Parameter (input)", "", "", true),
mWorker(*this)
{
    mVersion = "0.8";

    //Add plugin properties to property container
    mProperties.add(&mSBML);
    mProperties.add(&mOutputData);
    mProperties.add(&mParameter);
    mProperties.add(&mSelectionList);

    mProperties.add(&mStartTime);
    mProperties.add(&mEndTime);
    mProperties.add(&mNumberOfPointsPerSimulation);

    mHint = "Generate multiple data sets by incrementing one model parameters.";
    mDescription="";
    //The function below assigns property descriptions
    assignPropertyDescriptions();
}

ParameterScan::~ParameterScan()
{}

bool ParameterScan::isWorking() const
{
    return mWorker.isRunning();
}

unsigned char* ParameterScan::getManualAsPDF() const
{
    return pdf_doc;
}

unsigned int ParameterScan::getPDFManualByteSize()
{
    return sizeofPDF;
}

string ParameterScan::getImplementationLanguage()
{
    return ::getImplementationLanguage();
}

bool ParameterScan::resetPlugin()
{
    if(mWorker.isRunning())
    {
        return false;
    }

    mTerminate = false;
    return true;
}

bool ParameterScan::execute(bool inThread)
{
    try
    {
        RRPLOG(lInfo)<<"Executing the ParameterScan plugin";
        mWorker.start(inThread);
        return true;
    }
    catch(const Exception& ex)
    {
        RRPLOG(lError) << "There was a problem in the execute function of the ParameterScan plugin: " << ex.getMessage();
        throw(ex);
    }
    catch(...)
    {
        RRPLOG(lError) << "There was an unknown problem in the execute of the ParameterScan plugin.";
        throw("There was an unknown problem in the execute of the ParameterScan plugin.");
    }
}

// Plugin factory function
ParameterScan* plugins_cc createPlugin()
{
    //allocate a new object and return it
    return new ParameterScan();
}

const char* plugins_cc getImplementationLanguage()
{
    return "CPP";
}

void ParameterScan::assignPropertyDescriptions()
{
    stringstream s;

s << "Experimental data is used for input.";
mOutputData.setDescription(s.str());
s.str("");

s << "Property holding parameters to scan.";
mParameter.setDescription(s.str());
s.str("");

}

