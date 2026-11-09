//
// Timing comparison of the LLJit optimization levels (and MCJit) on numerical workloads other
// than the deterministic ODE simulation, on real BioModels SBML files. Currently this is a seeded
// Gillespie run. runWorkloadTiming runs any workload (a function of a loaded model) over the same
// variants, so adding a new workload only takes a new TEST that calls it.
//
// Built only with -DRR_BUILD_TIMING_TESTS=ON. By default the tests use the models in
// test/models/BioModelsTiming and skip themselves if the model folder does not exist. Run them
// with a Release build:
//   test_cxx_api_LLJitOptimizationLevelNumericsTimingTests
//
// Environment variables:
//   RR_TIMING_BIOMODELS_DIR  folder holding one subfolder per model, each with <id>_url.xml
//   RR_TIMING_MODELS         comma separated model ids (default for Gillespie is
//                            BIOMD0000000014,BIOMD0000000574,BIOMD0000000255)
//   RR_TIMING_REPS           recorded repetitions per variant (default 2)
//   RR_TIMING_MCJIT          set to 0 to leave MCJit out (default 1, MCJit is included)
//   RR_TIMING_GILLESPIE_TIME duration of the Gillespie simulation (default 100)
//   RR_TIMING_GILLESPIE_PTS  output points of the Gillespie simulation (default 10001)
//
// The checksum column shows a number derived from the result of the last run of each variant.
// For Gillespie (fixed seed) it should agree between variants to several digits; a difference
// means the variants computed different things.
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
#include <functional>
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
    };

    struct Samples {
        std::vector<double> loadMs;
        std::vector<double> workMs;
        double checksum = 0;
        std::string note;
    };

    // The workload gets a freshly loaded model and returns a checksum of its result.
    using Workload = std::function<double(RoadRunner &)>;

    double minOf(const std::vector<double> &v) {
        return *std::min_element(v.begin(), v.end());
    }

    void runWorkloadTiming(const std::string &title, const std::string &workDescription,
                           const std::string &defaultModels, int defaultReps,
                           const Workload &workload) {
        const std::string dir = envString("RR_TIMING_BIOMODELS_DIR",
                                          (getRoadRunnerTestDirectory() / "models" / "BioModelsTiming").string());
        if (!std::filesystem::exists(dir)) {
            GTEST_SKIP() << "model folder not found: " << dir << " (set RR_TIMING_BIOMODELS_DIR)";
        }
        const std::vector<std::string> models = split(
                envString("RR_TIMING_MODELS", defaultModels), ',');
        const int reps = envInt("RR_TIMING_REPS", defaultReps);

        std::vector<Variant> variants = {
                {"LLJit, NONE",       Config::LLJIT, Config::NONE},
                {"LLJit, LESS",       Config::LLJIT, Config::LESS},
                {"LLJit, DEFAULT",    Config::LLJIT, Config::DEFAULT},
                {"LLJit, AGGRESSIVE", Config::LLJIT, Config::AGGRESSIVE},
        };
        const size_t nLLJit = variants.size();
        if (envInt("RR_TIMING_MCJIT", 1) == 1) {
            variants.push_back({"MCJit (reference)", Config::MCJIT, Config::NONE});
        }

        // Load a tiny model with each backend first so LLVM start-up costs are not timed.
        for (int backend : {(int) Config::LLJIT, (int) Config::MCJIT}) {
            Config::resetToDefaults();
            Config::setValue(Config::LLVM_BACKEND, backend);
            RoadRunner warmUp(tinySBML);
        }

        std::cout << "\n=== " << title << " ===" << std::fixed << std::setprecision(1) << std::endl;
        int counter = 0;
        int done = 0;
        const int totalRuns = (int) (models.size() * variants.size() * reps);

        for (const std::string &model : models) {
            std::filesystem::path file = std::filesystem::path(dir) / model / (model + "_url.xml");
            ASSERT_TRUE(std::filesystem::exists(file)) << file.string();
            const std::string sbml = readFile(file);

            std::vector<Samples> samples(variants.size());
            int nSpecies = 0;
            int nReactions = 0;

            // MCJit (the last variant, when present) runs first in every rep. The LLJit levels
            // that follow rotate their starting point each rep.
            for (int rep = 0; rep < reps; ++rep) {
                std::vector<size_t> order;
                if (variants.size() > nLLJit) {
                    order.push_back(nLLJit);
                }
                for (size_t k = 0; k < nLLJit; ++k) {
                    order.push_back((k + rep) % nLLJit);
                }
                for (size_t v : order) {
                    Config::resetToDefaults();
                    Config::setValue(Config::LLVM_BACKEND, variants[v].backend);
                    Config::setValue(Config::LLJIT_OPTIMIZATION_LEVEL, variants[v].optLevel);
                    Config::setValue(Config::LLJIT_NUM_THREADS, 1);
                    Config::setValue(Config::LOADSBMLOPTIONS_RECOMPILE, true);
                    Config::setValue(Config::LLVM_SYMBOL_CACHE, false);

                    std::string uniqueSBML = makeUnique(sbml, ++counter);

                    Clock::time_point t0 = Clock::now();
                    RoadRunner rr(uniqueSBML);
                    Clock::time_point t1 = Clock::now();
                    nSpecies = rr.getNumberOfFloatingSpecies();
                    nReactions = rr.getNumberOfReactions();

                    double work = -1;
                    try {
                        Clock::time_point t2 = Clock::now();
                        samples[v].checksum = workload(rr);
                        work = elapsedMs(t2, Clock::now());
                    } catch (const std::exception &e) {
                        samples[v].note = std::string(" (failed: ") + e.what() + ")";
                    }

                    double load = elapsedMs(t0, t1);
                    samples[v].loadMs.push_back(load);
                    if (work >= 0) {
                        samples[v].workMs.push_back(work);
                    }
                    std::cout << "[" << ++done << "/" << totalRuns << "] " << model << ", "
                              << variants[v].label << ": load " << load << " ms";
                    if (work >= 0) {
                        std::cout << ", " << workDescription << " " << work << " ms";
                    }
                    std::cout << samples[v].note << std::endl;
                }
            }

            std::cout << "\n" << model << ": " << nSpecies << " floating species, " << nReactions
                      << " reactions; " << reps << " reps; work = " << workDescription << ", in ms\n"
                      << std::left << std::setw(24) << "variant" << std::right
                      << std::setw(12) << "load min" << std::setw(12) << "load med"
                      << std::setw(12) << "work min" << std::setw(12) << "work med"
                      << std::setw(20) << "checksum" << "\n";
            for (size_t v = 0; v < variants.size(); ++v) {
                std::cout << std::left << std::setw(24) << variants[v].label << std::right
                          << std::setw(12) << minOf(samples[v].loadMs)
                          << std::setw(12) << median(samples[v].loadMs);
                if (samples[v].workMs.empty()) {
                    std::cout << std::setw(12) << "n/a" << std::setw(12) << "n/a";
                } else {
                    std::cout << std::setw(12) << minOf(samples[v].workMs)
                              << std::setw(12) << median(samples[v].workMs);
                }
                std::cout << std::setw(20) << std::setprecision(6) << samples[v].checksum
                          << std::setprecision(1) << samples[v].note << "\n";
            }
            std::cout << std::endl;
        }
        Config::resetToDefaults();
    }
}

// A Gillespie run with a fixed seed, so every variant follows the same trajectory.
TEST(LLJitOptimizationLevelNumericsTimingTests, GillespieRun) {
    const double endTime = envInt("RR_TIMING_GILLESPIE_TIME", 100);
    const int points = envInt("RR_TIMING_GILLESPIE_PTS", 10001);
    runWorkloadTiming("Gillespie", "Gillespie simulation", "BIOMD0000000014,BIOMD0000000574,BIOMD0000000255", 2,
                      [&](RoadRunner &rr) {
        rr.setIntegrator("gillespie");
        rr.getIntegrator()->setValue("seed", 1234);
        const ls::DoubleMatrix *result = rr.simulate(0, endTime, points);
        double sum = 0;
        unsigned int last = result->RSize() - 1;
        for (unsigned int c = 1; c < result->CSize(); ++c) {
            sum += result->Element(last, c);
        }
        return sum;
    });
}
