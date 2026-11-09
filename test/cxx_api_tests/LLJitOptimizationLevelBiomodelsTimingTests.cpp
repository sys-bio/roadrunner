//
// Timing comparison of the LLJit optimization levels on real BioModels SBML files: how long
// each takes to load, how fast the generated code evaluates reaction rates, and how long a
// simulation takes.
//
// By default the test uses the models in test/models/BioModelsTiming; set RR_TIMING_BIOMODELS_DIR
// to use another folder. Run it with a Release build:
//   RR_TIMING_BIOMODELS_DIR=/path/to/final test_cxx_api_LLJitOptimizationLevelBiomodelsTimingTests
//
// Environment variables:
//   RR_TIMING_BIOMODELS_DIR  folder holding one subfolder per model, each with <id>_url.xml
//   RR_TIMING_MODELS         comma separated model ids
//                            (default BIOMD0000000014,BIOMD0000000574,BIOMD0000000255)
//   RR_TIMING_REPS           recorded repetitions per variant (default 2)
//   RR_TIMING_EVALS          getReactionRates calls timed per load (default 5000)
//   RR_TIMING_SIM_TIME       duration of the timed simulation (default 10)
//   RR_TIMING_SIM_STEPS      output intervals of the timed simulation (default 1000)
//   RR_TIMING_MCJIT          set to 0 to leave MCJit out of the comparison (default 1, included)
//   RR_TIMING_THREADS        LLJIT_NUM_THREADS used for every LLJit load (default 1; 0 compiles
//                            on the calling thread)
//   RR_TIMING_ORDER          comma separated list of NONE, LESS, DEFAULT, AGGRESSIVE, MCJIT. When
//                            set, each model is loaded once per entry in exactly that order
//                            (repeats allowed, no rotation); RR_TIMING_REPS and RR_TIMING_MCJIT
//                            are then ignored. For example NONE,NONE,NONE,NONE
//

#include "gtest/gtest.h"
#include "rrConfig.h"
#include "rrRoadRunner.h"
#include "RoadRunnerTestDirectory.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace rr;

namespace {

    using Clock = std::chrono::steady_clock;

    int envInt(const char *name, int fallback) {
        const char *v = std::getenv(name);
        return v ? std::atoi(v) : fallback;
    }

    std::string envString(const char *name, const std::string &fallback) {
        const char *v = std::getenv(name);
        return v ? v : fallback;
    }

    double elapsedMs(Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    }

    double median(std::vector<double> v) {
        std::sort(v.begin(), v.end());
        size_t n = v.size();
        return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    }

    std::vector<std::string> split(const std::string &s, char sep) {
        std::vector<std::string> out;
        std::istringstream in(s);
        std::string item;
        while (std::getline(in, item, sep)) {
            if (!item.empty()) {
                out.push_back(item);
            }
        }
        return out;
    }

    std::string readFile(const std::filesystem::path &path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("could not open " + path.string());
        }
        std::ostringstream ss;
        ss << in.rdbuf();
        return ss.str();
    }

    // Appends a suffix to the model id so that every load has a different md5 and cannot be
    // served from a cache.
    std::string makeUnique(std::string sbml, int counter) {
        size_t model = sbml.find("<model");
        size_t tagEnd = sbml.find('>', model);
        size_t idPos = sbml.find(" id=\"", model);
        if (model == std::string::npos || idPos == std::string::npos || idPos > tagEnd) {
            throw std::runtime_error("no id attribute found on the model element");
        }
        size_t idEnd = sbml.find('"', idPos + 5);
        sbml.insert(idEnd, "_run" + std::to_string(counter));
        return sbml;
    }

    const char *tinySBML =
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
            "<sbml xmlns=\"http://www.sbml.org/sbml/level3/version1/core\" level=\"3\" version=\"1\">\n"
            "<model id=\"tiny\">\n"
            "<listOfCompartments><compartment id=\"c\" spatialDimensions=\"3\" size=\"1\" constant=\"true\"/>"
            "</listOfCompartments>\n"
            "<listOfSpecies><species id=\"S\" compartment=\"c\" initialConcentration=\"1\""
            " hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/></listOfSpecies>\n"
            "<listOfParameters><parameter id=\"k\" value=\"0.1\" constant=\"true\"/></listOfParameters>\n"
            "<listOfReactions><reaction id=\"R\" reversible=\"false\" fast=\"false\">"
            "<listOfReactants><speciesReference species=\"S\" stoichiometry=\"1\" constant=\"true\"/></listOfReactants>"
            "<kineticLaw><math xmlns=\"http://www.w3.org/1998/Math/MathML\"><apply><times/><ci>k</ci><ci>S</ci>"
            "</apply></math></kineticLaw></reaction></listOfReactions>\n"
            "</model>\n</sbml>\n";

    struct Variant {
        std::string label;
        int backend;
        int optLevel;
        std::string token;
    };

    struct Samples {
        std::vector<double> loadMs;
        std::vector<double> evalUs;
        std::vector<double> simMs;
    };

    double minOf(const std::vector<double> &v) {
        return *std::min_element(v.begin(), v.end());
    }

    void printCell(const std::vector<double> &v, bool useMedian) {
        if (v.empty()) {
            std::cout << std::setw(12) << "n/a";
        } else {
            std::cout << std::setw(12) << (useMedian ? median(v) : minOf(v));
        }
    }
}

TEST(LLJitOptimizationLevelBiomodelsTimingTests, LoadTimeAndSimulationVersusOptLevel) {
    const std::string dir = envString("RR_TIMING_BIOMODELS_DIR",
                                          (getRoadRunnerTestDirectory() / "models" / "BioModelsTiming").string());
    if (dir.empty()) {
        GTEST_SKIP() << "RR_TIMING_BIOMODELS_DIR is not set";
    }
    const std::vector<std::string> models = split(
            envString("RR_TIMING_MODELS", "BIOMD0000000014,BIOMD0000000574,BIOMD0000000595,BIOMD0000000255"), ',');
    const int reps = envInt("RR_TIMING_REPS", 2);
    const int nEvals = envInt("RR_TIMING_EVALS", 5000);
    const int simTime = envInt("RR_TIMING_SIM_TIME", 1000);
    const int simSteps = envInt("RR_TIMING_SIM_STEPS", 10000);
    const int numThreads = envInt("RR_TIMING_THREADS", 0);

    const std::vector<Variant> variants = {
            {"LLJit, NONE",       Config::LLJIT, Config::NONE,       "NONE"},
            {"LLJit, LESS",       Config::LLJIT, Config::LESS,       "LESS"},
            {"LLJit, DEFAULT",    Config::LLJIT, Config::DEFAULT,    "DEFAULT"},
            {"LLJit, AGGRESSIVE", Config::LLJIT, Config::AGGRESSIVE, "AGGRESSIVE"},
            {"MCJit (reference)", Config::MCJIT, Config::NONE,       "MCJIT"},
    };

    // Each round is a list of variant indices, run in order. By default there is one round per
    // rep. MCJit (the last variant) runs first in every round and the LLJit levels that follow
    // rotate their starting point each round.
    const std::string orderSpec = envString("RR_TIMING_ORDER", "");
    std::vector<std::vector<size_t>> schedule;
    if (orderSpec.empty()) {
        const size_t nLLJit = variants.size() - 1;
        for (int rep = 0; rep < reps; ++rep) {
            std::vector<size_t> round;
            if (envInt("RR_TIMING_MCJIT", 1) == 1) {
                round.push_back(nLLJit);
            }
            for (size_t k = 0; k < nLLJit; ++k) {
                round.push_back((k + rep) % nLLJit);
            }
            schedule.push_back(round);
        }
    } else {
        std::vector<size_t> round;
        for (const std::string &token : split(orderSpec, ',')) {
            auto it = std::find_if(variants.begin(), variants.end(),
                                   [&](const Variant &x) { return x.token == token; });
            ASSERT_TRUE(it != variants.end()) << "unknown entry in RR_TIMING_ORDER: " << token;
            round.push_back((size_t) (it - variants.begin()));
        }
        schedule.push_back(round);
    }
    size_t runsPerModel = 0;
    for (const std::vector<size_t> &round : schedule) {
        runsPerModel += round.size();
    }

    // Load a tiny model with each backend first so LLVM start-up costs are not timed.
    std::vector<int> warmedBackends;
    for (const std::vector<size_t> &round : schedule) {
        for (size_t v : round) {
            int backend = variants[v].backend;
            if (std::find(warmedBackends.begin(), warmedBackends.end(), backend) == warmedBackends.end()) {
                warmedBackends.push_back(backend);
                Config::resetToDefaults();
                Config::setValue(Config::LLVM_BACKEND, backend);
                RoadRunner warmUp(tinySBML);
            }
        }
    }

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "LLJIT_NUM_THREADS = " << numThreads << std::endl;
    int counter = 0;
    int done = 0;
    const int totalRuns = (int) (models.size() * runsPerModel);

    for (const std::string &model : models) {
        std::filesystem::path file = std::filesystem::path(dir) / model / (model + "_url.xml");
        ASSERT_TRUE(std::filesystem::exists(file)) << file.string();
        const std::string sbml = readFile(file);

        std::vector<Samples> samples(variants.size());
        int nSpecies = 0;
        int nReactions = 0;

        for (const std::vector<size_t> &round : schedule) {
            for (size_t v : round) {
                Config::resetToDefaults();
                Config::setValue(Config::LLVM_BACKEND, variants[v].backend);
                Config::setValue(Config::LLJIT_OPTIMIZATION_LEVEL, variants[v].optLevel);
                Config::setValue(Config::LLJIT_NUM_THREADS, numThreads);
                Config::setValue(Config::LOADSBMLOPTIONS_RECOMPILE, true);
                Config::setValue(Config::LLVM_SYMBOL_CACHE, false);

                std::string uniqueSBML = makeUnique(sbml, ++counter);

                Clock::time_point t0 = Clock::now();
                RoadRunner rr(uniqueSBML);
                Clock::time_point t1 = Clock::now();
                nSpecies = rr.getNumberOfFloatingSpecies();
                nReactions = rr.getNumberOfReactions();

                double checksum = 0;
                for (int i = 0; i < nEvals; ++i) {
                    std::vector<double> rates = rr.getReactionRates();
                    checksum += rates.empty() ? 0.0 : rates[0];
                }
                Clock::time_point t2 = Clock::now();

                double sim = -1;
                std::string simNote;
                try {
                    Clock::time_point t3 = Clock::now();
                    rr.simulate(0, simTime, simSteps + 1);
                    sim = elapsedMs(t3, Clock::now());
                } catch (const std::exception &e) {
                    simNote = std::string(", simulate failed: ") + e.what();
                }

                double load = elapsedMs(t0, t1);
                double perEvalUs = elapsedMs(t1, t2) * 1000.0 / nEvals;
                samples[v].loadMs.push_back(load);
                samples[v].evalUs.push_back(perEvalUs);
                if (sim >= 0) {
                    samples[v].simMs.push_back(sim);
                }
                std::cout << "[" << ++done << "/" << totalRuns << "] " << model << ", " << variants[v].label
                          << ": load " << load << " ms, " << perEvalUs << " us per getReactionRates call";
                if (sim >= 0) {
                    std::cout << ", simulate " << sim << " ms";
                }
                std::cout << simNote << (std::isfinite(checksum) ? "" : " (non-finite rates)") << std::endl;
            }
        }

        std::cout << "\n" << model << ": " << nSpecies << " floating species, " << nReactions << " reactions; "
                  << (orderSpec.empty() ? std::to_string(reps) + " reps" : "order " + orderSpec) << "; " << nEvals << " getReactionRates calls per load; simulation of "
                  << simSteps << " steps over " << simTime << " time units\n"
                  << "load = RoadRunner constructor (parse + codegen + JIT compile), in ms. "
                  << "eval = time per getReactionRates call, in us. sim = time for one simulation, in ms.\n"
                  << std::left << std::setw(24) << "variant" << std::right
                  << std::setw(12) << "load min" << std::setw(12) << "load med"
                  << std::setw(12) << "eval min" << std::setw(12) << "eval med"
                  << std::setw(12) << "sim min" << std::setw(12) << "sim med" << "\n";
        for (size_t v = 0; v < variants.size(); ++v) {
            if (samples[v].loadMs.empty()) {
                continue;
            }
            std::cout << std::left << std::setw(24) << variants[v].label << std::right;
            printCell(samples[v].loadMs, false);
            printCell(samples[v].loadMs, true);
            printCell(samples[v].evalUs, false);
            printCell(samples[v].evalUs, true);
            printCell(samples[v].simMs, false);
            printCell(samples[v].simMs, true);
            std::cout << "\n";
        }
        std::cout << std::endl;
    }
    Config::resetToDefaults();
}
