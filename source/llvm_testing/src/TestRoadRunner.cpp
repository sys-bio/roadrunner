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
#include "rrSBMLReader.h"
#include "rrExecutableModel.h"

#include "conservation/ConservationExtension.h"
#include "conservation/ConservationDocumentPlugin.h"
#include "conservation/ConservedMoietyPlugin.h"
#include "conservation/ConservedMoietyConverter.h"


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
    dataOutputFolder = home + std::string("/tmp");
    std::string dummy;
    std::string logFileName;
    std::string settingsFileName;



    //Create a log file name
    createTestSuiteFileNameParts(caseNumber, ".log", dummy, logFileName, settingsFileName, dummy);

    //Create subfolder for data output
    dataOutputFolder = joinPath(dataOutputFolder, getTestSuiteSubFolderName(caseNumber));

    if(!createFolder(dataOutputFolder))
    {
        std::string msg("Failed creating output folder for data output: " + dataOutputFolder);
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
    std::string dummy;
    createTestSuiteFileNameParts(caseNumber, "-sbml-" + version + ".xml",
            modelFilePath, modelFileName, settingsFileName, dummy);

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
    std::string settingsOveride("");
    if (!simulation->LoadSettingsEx(settingsOveride))
    {
        rrLog(Logger::LOG_ERROR) << "Failed loading SBML model settings";
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
	options.reset_model = true;
    // TODO is this correct?

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
        rrLog(Logger::LOG_NOTICE) << "Test failed..\n";
    }
    else
    {
        rrLog(Logger::LOG_NOTICE) << "Test passed..\n";
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
        std::cout << "row " << r << ", [";
        for (int c = 0; c < data->CSize; ++c) {
            std::cout << data->Data[r*data->CSize + c] << ", ";
        }
        std::cout << "]" << std::endl;
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
        std::cout << "row " << r << ", [";
        for (int c = 0; c < data->CSize; ++c) {
            std::cout << data->Data[r*data->CSize + c] << ", ";
        }
        std::cout << "]" << std::endl;
    }
	*/
#endif
}



/*

bool RunTest(const std::string& version, int caseNumber)
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
        std::string settingsOveride("");
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
            clog<<"\t\tTest failed..\n";
        }
        else
        {
            clog<<"\t\tTest passed..\n";
        }
    }
    catch(rr::Exception& ex)
    {
        std::string error = ex.what();
        cerr<<"Case "<<caseNumber<<": Exception: "<<error<<std::endl;
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

std::string TestRoadRunner::read_uri(const std::string& uri)
{

    return SBMLReader::read(uri);

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
        std::cout << "caught exception " << ex.what() << std::endl;
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
        std::cout << "error: " << e.what() << std::endl;
    }
}


void TestRoadRunner::testCons1()
{
    ConservationPkgNamespaces *sbmlns = new ConservationPkgNamespaces(3,1,1);

    SBMLDocument doc(sbmlns);

    ConservationDocumentPlugin* docPlugin =
            dynamic_cast<ConservationDocumentPlugin*>(doc.getPlugin("conservation"));

    std::cout << "document plugin: " << docPlugin << std::endl;

    Model *m = doc.createModel("foo");

    Parameter *p = m->createParameter();

    ConservedMoietyPlugin *paramPlugin =
            dynamic_cast<ConservedMoietyPlugin*>(p->getPlugin("conservation"));

    std::cout << "parameter plugin: " << paramPlugin << std::endl;

    Species *s = m->createSpecies();

    ConservedMoietyPlugin *speciesPlugin =
            dynamic_cast<ConservedMoietyPlugin*>(s->getPlugin("conservation"));

    std::cout << "species plugin: " << speciesPlugin << std::endl;



    std::cout << "its all good" << std::endl;

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

    conv.convert();

    SBMLDocument *newDoc = conv.getDocument();

    ConservationDocumentPlugin* docPlugin =
            dynamic_cast<ConservationDocumentPlugin*>(newDoc->getPlugin(
                    "conservation"));

    std::cout << "document plugin: " << docPlugin << std::endl;

    libsbml::SBMLWriter writer;

    std::string base = removeExtension(fname);

    writer.writeSBML(conv.getLevelConvertedDocument(), base + ".l3v1.xml");

    writer.writeSBML(newDoc, base + ".moiety.xml");

    delete doc;

    std::cout << "its all good" << std::endl;
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

        std::cout << "param \'" << p->getId() << "\', conservedMoiety: "
                << ConservationExtension::getConservedMoiety(*p) << std::endl;
    }

    const ListOfSpecies *species = m->getListOfSpecies();

    for(int i = 0; i < species->size(); ++i)
    {
        const Species *s = species->get(i);

        std::cout << "species \'" << s->getId() << "\', conservedMoiety: "
                        << ConservationExtension::getConservedMoiety(*s) << std::endl;

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



    std::cout << "its all good" << std::endl;
}

void TestRoadRunner::testLogging(const std::string& logFileName)
{
    Logger::enableConsoleLogging(Logger::LOG_NOTICE);

    rrLog(Logger::LOG_NOTICE) << "console only notice";

    rrLog(Logger::LOG_NOTICE) << "setting logging to file: " << logFileName;

    Logger::enableFileLogging(logFileName, Logger::LOG_NOTICE);

    std::cout << "console and file logging:" << std::endl;

    std::cout << "log file name: " << Logger::getFileName() << std::endl;

    rrLog(Logger::LOG_FATAL) << "console and file: A fatal error";
    rrLog(Logger::LOG_CRITICAL) << "console and file: A critical error";
    rrLog(Logger::LOG_ERROR) << "console and file: An error";
    rrLog(Logger::LOG_WARNING) << "console and file: A warning. ";
    rrLog(Logger::LOG_NOTICE) << "console and file: A notice.";
    rrLog(Logger::LOG_INFORMATION) << "console and file: An informational message";
    rrLog(Logger::LOG_DEBUG) << "console and file: A debugging message.";
    rrLog(Logger::LOG_TRACE) << "console and file: A tracing message.";

    Logger::disableConsoleLogging();

    std::cout << "file only logging:" << std::endl;

    std::cout << "log file name: " << Logger::getFileName() << std::endl;

    rrLog(Logger::LOG_FATAL) << "file only: A fatal error";
    rrLog(Logger::LOG_CRITICAL) << "file only: A critical error";
    rrLog(Logger::LOG_ERROR) << "file only: An error";
    rrLog(Logger::LOG_WARNING) << "file only: A warning. ";
    rrLog(Logger::LOG_NOTICE) << "file only: A notice.";
    rrLog(Logger::LOG_INFORMATION) << "file only: An informational message";
    rrLog(Logger::LOG_DEBUG) << "file only: A debugging message.";
    rrLog(Logger::LOG_TRACE) << "file only: A tracing message.";

    std::cout << "no logging: " << std::endl;

    Logger::disableLogging();

    std::cout << "log file name: " << Logger::getFileName() << std::endl;

    rrLog(Logger::LOG_FATAL) << "no log: A fatal error";
    rrLog(Logger::LOG_CRITICAL) << "no log: A critical error";
    rrLog(Logger::LOG_ERROR) << "no log: An error";
    rrLog(Logger::LOG_WARNING) << "no log: A warning. ";
    rrLog(Logger::LOG_NOTICE) << "no log: A notice.";
    rrLog(Logger::LOG_INFORMATION) << "no log: An informational message";
    rrLog(Logger::LOG_DEBUG) << "no log: A debugging message.";
    rrLog(Logger::LOG_TRACE) << "no log: A tracing message.";

    Logger::enableConsoleLogging();

    std::cout << "console logging: " << std::endl;

    rrLog(Logger::LOG_FATAL) << "console logging: A fatal error";
    rrLog(Logger::LOG_CRITICAL) << "console logging: A critical error";
    rrLog(Logger::LOG_ERROR) << "console logging: An error";
    rrLog(Logger::LOG_WARNING) << "console logging: A warning. ";
    rrLog(Logger::LOG_NOTICE) << "console logging: A notice.";
    rrLog(Logger::LOG_INFORMATION) << "console logging: An informational message";
    rrLog(Logger::LOG_DEBUG) << "console logging: A debugging message.";
    rrLog(Logger::LOG_TRACE) << "console logging: A tracing message.";

    Logger::enableFileLogging(logFileName, Logger::LOG_NOTICE);

    std::cout << "console and file logging:" << std::endl;

    std::cout << "log file name: " << Logger::getFileName() << std::endl;

    rrLog(Logger::LOG_FATAL) << "console and file: A fatal error";
    rrLog(Logger::LOG_CRITICAL) << "console and file: A critical error";
    rrLog(Logger::LOG_ERROR) << "console and file: An error";
    rrLog(Logger::LOG_WARNING) << "console and file: A warning. ";
    rrLog(Logger::LOG_NOTICE) << "console and file: A notice.";
    rrLog(Logger::LOG_INFORMATION) << "console and file: An informational message";
    rrLog(Logger::LOG_DEBUG) << "console and file: A debugging message.";
    rrLog(Logger::LOG_TRACE) << "console and file: A tracing message.";

}

} /* namespace rr */


