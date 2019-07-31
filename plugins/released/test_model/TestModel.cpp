#pragma hdrstop
#include "rrRoadRunner.h"
#include "TestModel.h"
#include "rrException.h"
#include "rrLogger.h"
#include "rrUtils.h"
#include "telPluginManager.h"
#include "test_model_doc.h"

extern string theModel;
//---------------------------------------------------------------------------
TestModel::TestModel(PluginManager* manager)
:
CPPPlugin(  "TestModel",                 "Examples",    NULL,    manager),  //Construct Base
//Properties
mModel(                         "",                     "Model",                    "A SBML model"),
mTestData(                      TelluriumData(),        "TestData",                 "Simulated Data"),
mTestDataWithNoise(             TelluriumData(),        "TestDataWithNoise",        "Simulated Data With Noise"),
mSigma(                         3.e-6,                  "Sigma",                    "Sigma (<=> size of applied noise)")
{
    mVersion = "1.0.0";
    //Setup the plugins properties
    mProperties.add(&mModel);
    mProperties.add(&mTestData);
    mProperties.add(&mTestDataWithNoise);

    mHint ="Get access to a SBML model, and simulated data using the model.";
    mDescription="The TestModel plugin exposes properties representing a simple SBML model and simulated data using the model. The purpose of this plugin is to give a client easy access to a test model \
as well as data to test with. \
The TestModel plugin was developed at the University of Washington by Totte Karlsson, 2012-2014.";

    //Load the model from file here..
    try
    {
        mModel.setValue(theModel);
    }
    catch(const exception& ex)
    {
        Log(lError)<<"There was a problem in the TestModel plugin: "<<ex.what();
    }
}

TestModel::~TestModel()
{}

unsigned char* TestModel::getManualAsPDF() const
{
    return pdf_doc;
}

unsigned int TestModel::getPDFManualByteSize()
{
    return sizeofPDF;
}

bool TestModel::execute(bool inThread)
{
    rr::Logger::setLevel(rr::Logger::LOG_DEBUG);
    Log(lDebug)<<"Executing the TestModel plugin by J Kyle Medley and Totte Karlsson";
    Log(lDebug) << "Using SBML model: \n" << mModel.getValue();
    RoadRunner rr(mModel.getValue());

    rr::SimulateOptions opt;
    opt.start       = 0;
    opt.duration    = 10;
    opt.steps       = 14;

    rr.simulate(&opt);
    TelluriumData data;
    data.setData(*rr.getSimulationData());
    mTestData.setValue(data);

    //Add noise
    const PluginManager* PM = this->getPluginManager();
    Plugin* noise = PM->getPlugin("AddNoise");

    if(!noise)
    {
        stringstream msg;
        msg<<"The TestModel plugin dependes on the AddNoise plugin, which is not yet loaded.";

        throw(Exception(msg.str()));

    }
    mTestDataWithNoise.setValue(mTestData.getValue());

    noise->setPropertyValue("Sigma", mSigma.getValueHandle());
    noise->setPropertyValue("InputData", mTestDataWithNoise.getValueHandle());
    noise->execute();

    mTestDataWithNoise.setValue(noise->getPropertyValueHandle("InputData"));

    //Add weights
    addWeights();
    return true;
}

void TestModel::addWeights()
{
    TelluriumData &data = * (TelluriumData*) mTestDataWithNoise.getValueHandle();
    if(!data.hasWeights())
    {
        data.allocateWeights();
    }

    double sigma = mSigma;

    for(int r = 0; r < data.rSize(); r++)
    {
        for(int c = 0; c < data.cSize(); c++)
        {
            if( compareNoCase(data.getColumnName(c), "Time") == false )
            {
                double weight = data.getWeight(r,c);
                if(weight == 1)
                {
                    data.setWeight(r,c, weight * sigma * sigma);
                }
                else
                {
                    data.setWeight(r,c, weight + sigma * sigma);
                }
            }
        }
    }
}

void TestModel::assignPropertyDescriptions()
{
    stringstream s;

s << "The actual test model, in XML format.";
mModel.setDescription(s.str());
s.str("");

s << "Simulated data, using the TestModel as input and default RoadRunner Simulation values.";
mTestData.setDescription(s.str());
s.str("");

s << "Simulated data, with applied noise. ";
mTestDataWithNoise.setDescription(s.str());
s.str("");

}

// Plugin factory function
Plugin* plugins_cc createPlugin(void* manager)
{
    //allocate a new object and return it
    return new TestModel((PluginManager*) manager);
}

const char* plugins_cc getImplementationLanguage()
{
    return "CPP";
}

string  theModel =
"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
<sbml xmlns=\"http://www.sbml.org/sbml/level2/version4\" level=\"2\" version=\"4\">\n\
  <model metaid=\"_case00001\" id=\"case00001\" name=\"case00001\">\n\
    <listOfCompartments>\n\
      <compartment id=\"compartment\" name=\"compartment\" size=\"1\" units=\"volume\"/>\n\
    </listOfCompartments>\n\
    <listOfSpecies>\n\
      <species id=\"S1\" name=\"S1\" compartment=\"compartment\" initialAmount=\"0.00015\" substanceUnits=\"substance\"/>\n\
      <species id=\"S2\" name=\"S2\" compartment=\"compartment\" initialAmount=\"0\" substanceUnits=\"substance\"/>\n\
    </listOfSpecies>\n\
    <listOfParameters>\n\
      <parameter id=\"k1\" name=\"k1\" value=\"1\"/>\n\
    </listOfParameters>\n\
    <listOfReactions>\n\
      <reaction id=\"reaction1\" name=\"reaction1\" reversible=\"false\" fast=\"false\">\n\
        <listOfReactants>\n\
          <speciesReference species=\"S1\"/>\n\
        </listOfReactants>\n\
        <listOfProducts>\n\
          <speciesReference species=\"S2\"/>\n\
        </listOfProducts>\n\
        <kineticLaw>\n\
          <math xmlns=\"http://www.w3.org/1998/Math/MathML\">\n\
            <apply>\n\
              <times/>\n\
              <ci> compartment </ci>\n\
              <ci> k1 </ci>\n\
              <ci> S1 </ci>\n\
            </apply>\n\
          </math>\n\
        </kineticLaw>\n\
      </reaction>\n\
    </listOfReactions>\n\
  </model>\n\
</sbml>\n\
"
;

