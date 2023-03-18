#pragma hdrstop
#include "TestModel.h"
#include "telException.h"
#include "telLogger.h"
#include "telUtils.h"
#include "test_model_doc.h"

//#include "../add_noise/add_noise.h"


extern string theModel;
namespace testModel {
    rrc::THostInterface* gHostInterface = NULL;
    tlpc::TELHandle gPluginManager = NULL;
    //---------------------------------------------------------------------------
    TestModel::TestModel() :
        CPPPlugin("TestModel", "Examples"),//Construct Base
        //Properties
        mModel("", "Model", "A SBML model"),
        mTestData(TelluriumData(), "TestData", "Simulated Data"),
        mTestDataWithNoise(TelluriumData(), "TestDataWithNoise", "Simulated Data With Noise"),
        mSigma(3.e-6, "Sigma", "Sigma (<=> size of applied noise)"),
        mNumPoints(14, "NumPoints", "Number of points in the simulation."),
        mTimeEnd(10, "TimeEnd", "End time of the simulation."),
        mSeed(0, "Seed", "Random number seed for applying noise to data (0 to ignore)")
    {
        mVersion = "1.0.0";
        //Setup the plugins properties
        mProperties.add(&mModel);
        mProperties.add(&mTestData);
        mProperties.add(&mTestDataWithNoise);
        mProperties.add(&mSigma);
        mProperties.add(&mNumPoints);
        mProperties.add(&mTimeEnd);
        mProperties.add(&mSeed);

        mHint = "Get access to a SBML model, and simulated data using the model.";
        mDescription = "The TestModel plugin exposes properties representing a simple SBML model and simulated data using the model. The purpose of this plugin is to give a client easy access to a test model \
as well as data to test with. \
The TestModel plugin was developed at the University of Washington by Totte Karlsson, 2012-2014.";

        //Load the model from file here..
        try
        {
            mModel.setValue(theModel);
        }
        catch (const exception& ex)
        {
            RRPLOG(lError) << "There was a problem in the TestModel plugin: " << ex.what();
        }
    }

    TestModel::~TestModel()
    {}

    unsigned char* TestModel::getManualAsPDF() const
    {
        return pdf_doc;
    }

    size_t TestModel::getPDFManualByteSize()
    {
        return sizeofPDF;
    }

    bool TestModel::execute(bool inThread)
    {
        //tlp::Logger::setLevel(tlp::Logger::LOG_DEBUG);
        RRPLOG(lInfo) << "Executing the TestModel plugin by J Kyle Medley and Totte Karlsson";
        RRPLOG(lInfo) << "Using SBML model: \n" << mModel.getValue();

        rrc::RRHandle rrHandle = gHostInterface->createRRInstance();        //start
        rrc::RRCDataPtr result = NULL;
        gHostInterface->loadSBML(rrHandle, (mModel.getValue()).c_str());

        result = gHostInterface->simulateEx(rrHandle, 0, mTimeEnd.getValue(), mNumPoints.getValue());

        //convert rrcptr to matrix <double> 
        Matrix <double> output(result->RSize, result->CSize);

        for (int i = 0; i < result->RSize; ++i) {       //costly, 
            for (int j = 0; j < result->CSize; ++j) {
                output(i, j) = (result->Data)[i * result->CSize + j];
            }
        }

        std::vector<string> selections;

        for (int i = 0; i < result->CSize; ++i) {
            selections.push_back((result->ColumnHeaders)[i]);
        }

        output.setColNames(selections.begin(), selections.end());

        TelluriumData data;
        data.setData(output);       //end 
        mTestData.setValue(data);

        mTestDataWithNoise.setValue(mTestData.getValue());

        Plugin* noise = (Plugin*)gHostInterface->getPlugin(gPluginManager, "tel_add_noise");
        noise->setPropertyValue("Sigma", mSigma.getValueHandle());
        noise->setPropertyValue("InputData", mTestDataWithNoise.getValueHandle());
        noise->setPropertyValue("Seed", mSeed.getValueHandle());
        noise->execute();


        mTestDataWithNoise.setValue(noise->getPropertyValueHandle("InputData"));

        //Add weights
        addWeights();
        return true;
    }

    void TestModel::addWeights()
    {
        TelluriumData& data = *(TelluriumData*)mTestDataWithNoise.getValueHandle();
        if (!data.hasWeights())
        {
            data.allocateWeights();
        }

        double sigma = mSigma;

        for (int r = 0; r < data.rSize(); r++)
        {
            for (int c = 0; c < data.cSize(); c++)
            {
                if (compareNoCase(data.getColumnName(c), "Time") == false)
                {
                    double weight = data.getWeight(r, c);
                    if (weight == 1)
                    {
                        data.setWeight(r, c, weight * sigma * sigma);
                    }
                    else
                    {
                        data.setWeight(r, c, weight + sigma * sigma);
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


    #ifdef EXPORT_TEST_MODEL
    // Plugin factory function
    TLP_DS TestModel* plugins_cc createPlugin()
    {
        //allocate a new object and return it
        return new TestModel;
    }

    TLP_DS const char* plugins_cc getImplementationLanguage()
    {
        return "CPP";
    }

    TLP_DS void plugins_cc setHostInterface(rrc::THostInterface* _hostInterface) {
        gHostInterface = _hostInterface;
    }

    TLP_DS void plugins_cc setPluginManager(tlpc::TELHandle manager) {
        gPluginManager = manager;
    }
    #endif 


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
";

POCO_BEGIN_MANIFEST(tlp::Plugin)
	POCO_EXPORT_CLASS(testModel::TestModel)
POCO_END_MANIFEST



