/*
 * TestRoadRunner.cpp
 *
 *  Created on: Jul 24, 2013
 *      Author: andy
 */
#pragma hdrstop
#include "TestRoadRunner.h"
#include "rrUtils.h"
#include "rrException.h"
#include "rrLogger.h"
#include "rrc_api.h"
#include "rrSBMLReader.h"
#include "rrExecutableModel.h"

#include "conservation/ConservationExtension.h"
#include "conservation/ConservationDocumentPlugin.h"
#include "conservation/ConservedMoietyPlugin.h"
#include "conservation/ConservedMoietyConverter.h"

using namespace std;
using namespace libsbml;
using namespace rr::conservation;

namespace rr
{


TestRoadRunner::~TestRoadRunner()
{
    delete simulation;
    delete rr;
}

TestRoadRunner::TestRoadRunner(const std::string& version, int caseNumber) :
                version(version), caseNumber(caseNumber), rr(0), simulation(0)
{
    //fileName = getModelFileName(version, caseNumber);

    home = getenv("HOME");
    dataOutputFolder = home + string("/tmp");
    string dummy;
    string logFileName;
    string settingsFileName;



    //Create a log file name
    createTestSuiteFileNameParts(caseNumber, ".log", dummy, logFileName, settingsFileName);

    //Create subfolder for data output
    dataOutputFolder = joinPath(dataOutputFolder, getTestSuiteSubFolderName(caseNumber));

    if(!createFolder(dataOutputFolder))
    {
        string msg("Failed creating output folder for data output: " + dataOutputFolder);
        throw(Exception(msg));
    }
}

bool TestRoadRunner::test(const std::string& compiler)
{
    loadSBML(compiler);

    simulate();

    return true;
}

void TestRoadRunner::loadSBML(const std::string& compiler)
{
    rr = new RoadRunner(compiler, dataOutputFolder, home + "/local/rr_support");

    simulation = new TestSuiteModelSimulation(dataOutputFolder);

    simulation->UseEngine(rr);

    //Read SBML models.....
    modelFilePath = home + "/src/sbml_test/cases/semantic";

    simulation->SetCaseNumber(caseNumber);
    createTestSuiteFileNameParts(caseNumber, "-sbml-" + version + ".xml",
            modelFilePath, modelFileName, settingsFileName);

    //The following will load and compile and simulate the sbml model in the file
    simulation->SetModelFilePath(modelFilePath);
    simulation->SetModelFileName(modelFileName);
    simulation->ReCompileIfDllExists(true);
    simulation->CopyFilesToOutputFolder();
    //setTempFolder(gRR, simulation.GetDataOutputFolder().c_str());
    //setComputeAndAssignConservationLaws(gRR, false);

    //rr->loadSBMLFromFile(fileName);

    if (!simulation->LoadSBMLFromFile())
    {
        throw Exception("Failed loading sbml from file");
    }

    //Then read settings file if it exists..
    string settingsOveride("");
    if (!simulation->LoadSettingsEx(settingsOveride))
    {
        Log(Logger::LOG_ERROR) << "Failed loading SBML model settings";
        throw Exception("Failed loading SBML model settings");
    }

    rr->setConservedMoietyAnalysis(false);
}

void TestRoadRunner::simulate()
{
    SimulateOptions options;
    options.start = rr->getSimulateOptions().start;
    options.duration = rr->getSimulateOptions().duration;
    options.steps = rr->getSimulateOptions().steps;
    options.flags = options.flags | SimulateOptions::RESET_MODEL;

    if (!rr->simulate(&options))
    {
        throw Exception("Simulation Failed");
    }
}

void TestRoadRunner::saveResult()
{
    if (!simulation->SaveResult())
    {
        //Failed to save data
        throw Exception("Failed saving result");
    }
}

void TestRoadRunner::compareReference()
{
    if (!simulation->LoadReferenceData())
    {
        throw Exception("Failed Loading reference data");
    }

    simulation->CreateErrorData();
    bool result = simulation->Pass();
    simulation->SaveAllData();
    simulation->SaveModelAsXML(dataOutputFolder);
    if (!result)
    {
        Log(Logger::LOG_NOTICE) << "Test failed.\n";
    }
    else
    {
        Log(Logger::LOG_NOTICE) << "Test passed.\n";
    }
}

void TestRoadRunner::test2()
{
#ifndef _WIN32

	/*
    rrc::RRHandle rr = rrc::createRRInstance();

    rrc::loadSBMLFromFile(rr, "/home/andy/src/sbml_test/cases/semantic/00001/00001-sbml-l3v1.xml");

    //rrc::setTimeStart(rr, 0);
    //rrc::setTimeEnd(rr, 20);
    //rrc::setNumPoints(rr, 100);

    rrc::RRCDataPtr data = rrc::simulateEx(rr, 0, 20, 100);

    std::cout << "columns: " << data->CSize << ", rows: " << data->RSize << std::endl;

    for (int r = 0; r < data->RSize; ++r) {
        cout << "row " << r << ", [";
        for (int c = 0; c < data->CSize; ++c) {
            cout << data->Data[r*data->CSize + c] << ", ";
        }
        cout << "]" << std::endl;
    }
	*/
#endif
}


void TestRoadRunner::test3()
{
#ifndef _WIN32
	/*

    rrc::RRHandle rr = rrc::createRRInstance();

    rrc::loadSBMLFromFile(rr, "/Users/andy/Desktop/Feedback.xml");

    rrc::setTimeStart(rr, 0);
    rrc::setTimeEnd(rr, 20);
    rrc::setNumPoints(rr, 400);

    rrc::RRCDataPtr data = rrc::simulate(rr);

    std::cout << "columns: " << data->CSize << ", rows: " << data->RSize << std::endl;

    for (int r = 0; r < data->RSize; ++r) {
        cout << "row " << r << ", [";
        for (int c = 0; c < data->CSize; ++c) {
            cout << data->Data[r*data->CSize + c] << ", ";
        }
        cout << "]" << std::endl;
    }
	*/
#endif
}



/*

bool RunTest(const string& version, int caseNumber)
{
    bool result = false;
    RRHandle gRR = 0;

    //Create instance..
    gRR = createRRInstanceEx(gTempFolder.c_str());

    if(gDebug && gRR)
    {
        enableLoggingToConsole();
        setLogLevel("Debug5");
    }
    else
    {
        setLogLevel("Error");
    }

    //Setup environment
    setTempFolder(gRR, gTempFolder.c_str());

    if(!gRR)
    {
        return false;
    }

    try
    {




        //Then read settings file if it exists..
        string settingsOveride("");
        if(!simulation.LoadSettings(settingsOveride))
        {
            throw("Failed loading simulation settings");
        }

        //Then Simulate model
        if(!simulation.Simulate())
        {
            throw("Failed running simulation");
        }

        //Write result
        if(!simulation.SaveResult())
        {
            //Failed to save data
            throw("Failed saving result");
        }

        if(!simulation.LoadReferenceData())
        {
            throw("Failed Loading reference data");
        }

        simulation.CreateErrorData();
        result = simulation.Pass();
        simulation.SaveAllData();
        simulation.SaveModelAsXML(dataOutputFolder);
        if(!result)
        {
            clog<<"\t\tTest failed.\n";
        }
        else
        {
            clog<<"\t\tTest passed.\n";
        }
    }
    catch(rr::Exception& ex)
    {
        string error = ex.what();
        cerr<<"Case "<<caseNumber<<": Exception: "<<error<<endl;
        result = false;;
    }

    // done with rr
    freeRRInstance(gRR);
    return result;
}

 */

SelectionRecord TestRoadRunner::testsel(const std::string& str)
{
    return SelectionRecord(str);
}

std::string TestRoadRunner::read_uri(const std::string& uri, std::string& filename)
{

    return SBMLReader::read(uri, filename);

    /*


    try
    {
        Poco::Net::HTTPStreamFactory::registerFactory();

        Poco::URIStreamOpener &opener = Poco::URIStreamOpener::defaultOpener();


        std::istream* stream = opener.open(uri);

        std::istreambuf_iterator<char> eos;
        std::string s(std::istreambuf_iterator<char>(*stream), eos);

        return s;

    }
    catch(std::exception& ex)
    {
        cout << "caught exception " << ex.what() << endl;
        return ex.what();
    }

     */
}


void TestRoadRunner::steadyState(const std::string& uri)
{
    Logger::setLevel(Logger::LOG_DEBUG);
    RoadRunner r;

    r.load(uri);

    r.steadyState();
}

void TestRoadRunner::testLoad(const std::string& uri)
{
    try
    {
        Logger::setLevel(Logger::LOG_DEBUG);

        //std::string sbml = SBMLReader::read(uri);


        RoadRunner r;

        r.load(uri);

        r.steadyState();
    }
    catch(std::exception& e)
    {
        cout << "error: " << e.what() << std::endl;
    }
}


void TestRoadRunner::testCons1()
{
    ConservationPkgNamespaces *sbmlns = new ConservationPkgNamespaces(3,1,1);

    SBMLDocument doc(sbmlns);

    ConservationDocumentPlugin* docPlugin =
            dynamic_cast<ConservationDocumentPlugin*>(doc.getPlugin("conservation"));

    cout << "document plugin: " << docPlugin << endl;

    Model *m = doc.createModel("foo");

    Parameter *p = m->createParameter();

    ConservedMoietyPlugin *paramPlugin =
            dynamic_cast<ConservedMoietyPlugin*>(p->getPlugin("conservation"));

    cout << "parameter plugin: " << paramPlugin << endl;

    Species *s = m->createSpecies();

    ConservedMoietyPlugin *speciesPlugin =
            dynamic_cast<ConservedMoietyPlugin*>(s->getPlugin("conservation"));

    cout << "species plugin: " << speciesPlugin << endl;



    cout << "its all good" << endl;

}

std::string removeExtension(const std::string& filename)
{
    size_t lastdot = filename.find_last_of(".");
    if (lastdot == std::string::npos) return filename;
    return filename.substr(0, lastdot);
}

void TestRoadRunner::testCons2(const std::string& fname)
{

    Logger::enableConsoleLogging(Logger::LOG_DEBUG);
    //const char* fname = "/Users/andy/src/sbml_test/cases/semantic/00001/00001-sbml-l2v4.xml";

    libsbml::SBMLReader reader;

    SBMLDocument *doc = reader.readSBML(fname);

    ConservedMoietyConverter conv;

    conv.setDocument(doc);

    int result = conv.convert();

    SBMLDocument *newDoc = conv.getDocument();

    ConservationDocumentPlugin* docPlugin =
            dynamic_cast<ConservationDocumentPlugin*>(newDoc->getPlugin(
                    "conservation"));

    cout << "document plugin: " << docPlugin << endl;

    libsbml::SBMLWriter writer;

    string base = removeExtension(fname);

    writer.writeSBML(conv.getLevelConvertedDocument(), base + ".l3v1.xml");

    writer.writeSBML(newDoc, base + ".moiety.xml");

    delete doc;

    cout << "its all good" << endl;
}

void TestRoadRunner::testRead(const std::string &fname)
{
    Logger::enableConsoleLogging(Logger::LOG_DEBUG);
    //const char* fname = "/Users/andy/src/sbml_test/cases/semantic/00001/00001-sbml-l2v4.xml";

    libsbml::SBMLReader reader;

    SBMLDocument *doc = reader.readSBML(fname);


    Model *m = doc->getModel();

    const ListOfParameters *params = m->getListOfParameters();

    for(int i = 0; i < params->size(); ++i)
    {
        const Parameter *p = params->get(i);

        cout << "param \'" << p->getId() << "\', conservedMoiety: "
                << ConservationExtension::getConservedMoiety(*p) << endl;
    }

    const ListOfSpecies *species = m->getListOfSpecies();

    for(int i = 0; i < species->size(); ++i)
    {
        const Species *s = species->get(i);

        cout << "species \'" << s->getId() << "\', conservedMoiety: "
                        << ConservationExtension::getConservedMoiety(*s) << endl;

    }


    delete doc;

    Logger::setLevel(Logger::LOG_TRACE);

    RoadRunner r;

    r.load(fname, 0);


    rr::ExecutableModel *model = r.getModel();

    int len = model->getNumIndFloatingSpecies();
    double *buffer = new double[len];

    model->getFloatingSpeciesAmountRates(len, 0, buffer);

    delete[] buffer;



    cout << "its all good" << endl;
}

void TestRoadRunner::testLogging(const std::string& logFileName)
{
    Logger::enableConsoleLogging(Logger::LOG_NOTICE);

    Log(Logger::LOG_NOTICE) << "console only notice";

    Log(Logger::LOG_NOTICE) << "setting logging to file: " << logFileName;

    Logger::enableFileLogging(logFileName, Logger::LOG_NOTICE);

    cout << "console and file logging:" << endl;

    cout << "log file name: " << Logger::getFileName() << endl;

    Log(Logger::LOG_FATAL) << "console and file: A fatal error";
    Log(Logger::LOG_CRITICAL) << "console and file: A critical error";
    Log(Logger::LOG_ERROR) << "console and file: An error";
    Log(Logger::LOG_WARNING) << "console and file: A warning. ";
    Log(Logger::LOG_NOTICE) << "console and file: A notice.";
    Log(Logger::LOG_INFORMATION) << "console and file: An informational message";
    Log(Logger::LOG_DEBUG) << "console and file: A debugging message.";
    Log(Logger::LOG_TRACE) << "console and file: A tracing message.";

    Logger::disableConsoleLogging();

    cout << "file only logging:" << endl;

    cout << "log file name: " << Logger::getFileName() << endl;

    Log(Logger::LOG_FATAL) << "file only: A fatal error";
    Log(Logger::LOG_CRITICAL) << "file only: A critical error";
    Log(Logger::LOG_ERROR) << "file only: An error";
    Log(Logger::LOG_WARNING) << "file only: A warning. ";
    Log(Logger::LOG_NOTICE) << "file only: A notice.";
    Log(Logger::LOG_INFORMATION) << "file only: An informational message";
    Log(Logger::LOG_DEBUG) << "file only: A debugging message.";
    Log(Logger::LOG_TRACE) << "file only: A tracing message.";

    cout << "no logging: " << endl;

    Logger::disableLogging();

    cout << "log file name: " << Logger::getFileName() << endl;

    Log(Logger::LOG_FATAL) << "no log: A fatal error";
    Log(Logger::LOG_CRITICAL) << "no log: A critical error";
    Log(Logger::LOG_ERROR) << "no log: An error";
    Log(Logger::LOG_WARNING) << "no log: A warning. ";
    Log(Logger::LOG_NOTICE) << "no log: A notice.";
    Log(Logger::LOG_INFORMATION) << "no log: An informational message";
    Log(Logger::LOG_DEBUG) << "no log: A debugging message.";
    Log(Logger::LOG_TRACE) << "no log: A tracing message.";

    Logger::enableConsoleLogging();

    cout << "console logging: " << endl;

    Log(Logger::LOG_FATAL) << "console logging: A fatal error";
    Log(Logger::LOG_CRITICAL) << "console logging: A critical error";
    Log(Logger::LOG_ERROR) << "console logging: An error";
    Log(Logger::LOG_WARNING) << "console logging: A warning. ";
    Log(Logger::LOG_NOTICE) << "console logging: A notice.";
    Log(Logger::LOG_INFORMATION) << "console logging: An informational message";
    Log(Logger::LOG_DEBUG) << "console logging: A debugging message.";
    Log(Logger::LOG_TRACE) << "console logging: A tracing message.";

    Logger::enableFileLogging(logFileName, Logger::LOG_NOTICE);

    cout << "console and file logging:" << endl;

    cout << "log file name: " << Logger::getFileName() << endl;

    Log(Logger::LOG_FATAL) << "console and file: A fatal error";
    Log(Logger::LOG_CRITICAL) << "console and file: A critical error";
    Log(Logger::LOG_ERROR) << "console and file: An error";
    Log(Logger::LOG_WARNING) << "console and file: A warning. ";
    Log(Logger::LOG_NOTICE) << "console and file: A notice.";
    Log(Logger::LOG_INFORMATION) << "console and file: An informational message";
    Log(Logger::LOG_DEBUG) << "console and file: A debugging message.";
    Log(Logger::LOG_TRACE) << "console and file: A tracing message.";

}

#ifndef _WIN32
void TestRoadRunner::test_fs75()
{
    const char* src = "/Users/andy/fs75.xml";

    rrc::RRHandle r = rrc::createRRInstance();

    rrc::loadSBMLFromFile(r, src);

    double conc[] = {1.0, 2.0, 3.0};

    rrc::RRVector concVec;

    concVec.Count = 3;
    concVec.Data = conc;

    rrc::RRVectorPtr res = rrc::getReactionRatesEx(r, &concVec);

    for (int i = 0; i < res->Count; ++i)
    {
        cout << "index " << i << ": " << res->Data[i] << endl;
    }
}

void TestRoadRunner::test_fs74()
{
    rrc::RRHandle r = rrc::createRRInstance();

    rrc::loadSBMLFromFile(r, "");

    cout << "error:: " << rrc::getLastError() << endl;

}

void TestRoadRunner::test_fs73()
{
    const char* src = "/Users/andy/fs75.xml";

    rrc::RRHandle r = rrc::createRRInstance();

    rrc::loadSBMLFromFile(r, src);

    rrc::setFloatingSpeciesByIndex(r, 1, 3.14);

    double result = 0;

    rrc::getFloatingSpeciesByIndex(r, 1, &result);

    cout << "result: " << result << endl;
}
#else

void TestRoadRunner::test_fs75() {}
void TestRoadRunner::test_fs74() {}
void TestRoadRunner::test_fs73() {}

#endif

} /* namespace rr */


