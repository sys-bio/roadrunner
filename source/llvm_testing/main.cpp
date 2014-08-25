#pragma hdrstop

#include "TestEvalModel.h"
#include "TestRoadRunner.h"
#include "GetBoundarySpeciesAmountTest.h"
#include "TestEvalInitialConditions.h"
#include "TestGetSetValues.h"
#include "TestCapabilities.h"
#include "TestEvalReactionRates.h"
#include "rrConfig.h"
#include "rrUtils.h"

#include "SBMLValidator.h"

#include "ConfigurableTest.h"

#include "rrRoadRunner.h"

#include "Integrator.h"

#include "rrLogger.h"

#include "TestVariant.h"

#include <sbml/SBMLDocument.h>
#include <sbml/Model.h>
#include <sbml/SBMLReader.h>
#include <time.h>
#include <stdio.h>
#include <cmath>



using namespace std;
using namespace rr;



/**
 * perform a stochastic ensemble calculation.
 *
 *   :param: r an existing roadrunner obj, loaded with a mode.
 *   :param: n how many ensembles to run.
 *   :param: seed the seed value, use None to continue with previous
 *           random number stream.
 *   :param: start start time
 *   :param: stop stop time
 *   :param: ntps how many points display in the result.
 */
DoubleMatrix ensemble(RoadRunner &r, int n, unsigned long seed, double start, double stop, int npts) {

    // set the seed value of the integrator
    Integrator *itg = r.getIntegrator(SimulateOptions::GILLESPIE);
    itg->setValue("seed", seed);

    // setup the simulation
    // create a sim opt obj, loads itself with
    // default values in ctor.
    SimulateOptions o;
    o.start = start;
    o.duration = stop;
    o.steps = npts;

    // we should reset the model each time we simulate,
    // set the RESET_MODEL bit.
    o.flags |= SimulateOptions::RESET_MODEL;

    o.integrator = SimulateOptions::GILLESPIE;

    // make a result var
    DoubleMatrix result(npts+1, n+1);

    for (int i = 0; i < n; ++i) {
        const DoubleMatrix &sim = *r.simulate(&o);

        // copy result data colum
        for (int row = 0; row < npts+1; ++row) {
            result(row, i+1) = sim(row, 1);
        }
    }

    // copy result time column
    const DoubleMatrix &sim = *r.getSimulationData();

    // copy result data colum
    for (int row = 0; row < npts+1; ++row) {
        result(row, 0) = sim(row, 0);
    }

    return result;
}


void run_ensemble(const char* fname, unsigned long seed)
{
    try {
        RoadRunner r(fname);

        DoubleMatrix result = ensemble(r, 5, seed, 0, 10, 150);

        cout.precision(6);

        for(int row = 0; row < result.numRows(); ++row) {
            for(int col = 0; col < result.numCols(); ++col) {
                cout << result(row, col) << ", ";
            }
            cout << endl;
        }
    } catch (std::exception& e) {
        cout << "Error running ensemble: " << e.what() << endl;
    }
}

int ensemble_test(int argc, char* argv[])
{
    if (argc < 4)
    {
        cout << "usage: llvm_testing ensemble fname num_ensembles" << endl;
        return -1;
    }

    run_ensemble(argv[2], atol(argv[3]));

    return 0;
}

int path_test(int argc, char* argv[])
{
    try
    {
        std::string cmpTest = joinPath("..", "compilers", "tcc", "tcc.exe");
        cout << "comp test: " << cmpTest << endl;

        cout << "default tmp dir: " << getTempDir() << endl;

        Logger::enableFileLogging();
        std::string defLogFile = Logger::getFileName();

        cout << "default log file path: " << defLogFile << endl;

        if (argc >= 3) {
            cout << "setting config temp dir to: " << argv[2] << endl;
            Config::setValue(Config::TEMP_DIR_PATH, std::string(argv[2]));
        }

        if (argc >= 4) {
            cout << "setting config log file path to " << argv[3] << endl;
            Config::setValue(Config::LOGGER_LOG_FILE_PATH, std::string(argv[3]));
        }

        cout << "getTempDir: " << getTempDir() << endl;

        Logger::enableFileLogging();

        cout << "log file: " << Logger::getFileName() << endl;

        Log(Logger::LOG_NOTICE) << "log messsage";
    }
    catch (std::exception& e)
    {
        cout << "Fatal error: " << e.what() << endl;
    }

    return 0;
}

int jacobian_test(int argc, char* argv[])
{
    try
    {
        if (argc < 3)
        {
            cout << "usage: llvm_testing jacobian fname";
        }

        RoadRunner r(argv[2]);

        cout << "amount mode" << endl;
        Config::setValue(Config::ROADRUNNER_JACOBIAN_MODE, (unsigned)Config::ROADRUNNER_JACOBIAN_MODE_AMOUNTS);

        ls::DoubleMatrix mat = r.getFullJacobian();

        cout << "amount mode full jabobian: " << endl;

        cout << mat << endl;

        mat = r.getReducedJacobian();

        cout << "amount mode reduced jabobian: " << endl;

        cout << mat << endl;

        cout << "concentration mode" << endl;

        Config::setValue(Config::ROADRUNNER_JACOBIAN_MODE, (unsigned)Config::ROADRUNNER_JACOBIAN_MODE_CONCENTRATIONS);

        mat = r.getFullJacobian();

        cout << "concentration mode full jabobian: " << endl;

        cout << mat << endl;

        mat = r.getReducedJacobian();

        cout << "concentration mode reduced jabobian: " << endl;

        cout << mat << endl;

    }
    catch(std::exception &e)
    {
        cout << e.what() << endl;
    }
    return 0;
}


int cm_1_test(int argc, char* argv[])
{
    try {
        if (argc < 3)
        {
            cout << "usage: llvm_testing jacobian fname";
        }

        RoadRunner r(argv[2]);

        r.setConservedMoietyAnalysis(true);

        ExecutableModel *m = r.getModel();

        int ncm = m->getNumConservedMoieties();

        cout << "num conserved moieties: " << ncm << endl;

        double val;

        for (int i = 0; i < ncm; ++i) {
            std::string name = m->getConservedMoietyId(i);
            m->getConservedMoietyValues(1, &i, &val);

            cout << "cm name: " << name << ", value: " << val << endl;
        }


    } catch (std::exception& e) {
        cout << "Error running ensemble: " << e.what() << endl;
    }

    return 0;

}

int reset_test(int argc, char* argv[])
{
    try {
        if (argc < 4)
        {
            cout << "usage: llvm_testing reset fname what";
            return -1;
        }

        RoadRunner r(argv[2]);

        r.setConservedMoietyAnalysis(true);

        std::string what(argv[3]);

        uint opt = 0;

        if(what.find("GLOBAL_PARAMETER") != string::npos)
        {
            opt |= SelectionRecord::GLOBAL_PARAMETER;
        }

        if(what.find("TIME") != string::npos)
        {
            opt |= SelectionRecord::TIME;
        }

        if(what.find("FLOATING") != string::npos)
        {
            opt |= SelectionRecord::FLOATING;
        }

        if(what.find("COMPARTMENT") != string::npos)
        {
            opt |= SelectionRecord::COMPARTMENT;
        }

        if(what.find("RATE") != string::npos)
        {
            opt |= SelectionRecord::RATE;
        }

        if(what.find("ALL") != string::npos)
        {
            opt |= SelectionRecord::ALL;
        }

        r.reset(opt);

        return opt;
    }
    catch(std::exception& e) {
        cout << e.what();
    }
    return -1;
}


int validate_test(int argc, char* argv[])
{
    try {
        if (argc < 3)
        {
            cout << "usage: llvm_testing validate fname";
            return -1;
        }


        std::string errors = rr::validateSBML(argv[2], (rr::ValidateSBML)0);

        cout << errors << endl;

    }
    catch(std::exception& e) {
        cout << e.what();
    }
    return -1;
}


int main(int argc, char* argv[])
{
    if (argc < 2) {
        cout << "usage: llvm_testing test" << endl;
        return -1;
    }

    if (strcmp("ensemble", argv[1]) == 0) {
        return ensemble_test(argc, argv);
    }

    if (strcmp("jacobian", argv[1]) == 0) {
        return jacobian_test(argc, argv);
    }

    if (strcmp("cm_1", argv[1]) == 0) {
        return cm_1_test(argc, argv);
    }

    if(strcmp("path", argv[1]) == 0) {
        return path_test(argc, argv);
    }

    if(strcmp("reset", argv[1]) == 0) {
        return reset_test(argc, argv);
    }

    if(strcmp("validate", argv[1]) == 0) {
        return validate_test(argc, argv);
    }

    cout << "error, invalid test name: " << argv[1] << endl;
    return -1;
}






