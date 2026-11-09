//
// Timing comparison of LLJIT_NUM_THREADS when loading a single model.
//
// The test is DISABLED_ so a normal test run skips it. To run it:
//   test_cxx_api_LLJitCompileThreadsTimingTests --gtest_also_run_disabled_tests
//
// Optional environment variables:
//   RR_TIMING_REACTIONS  number of reactions in the generated model (default 200)
//   RR_TIMING_SPECIES    number of species in the generated model (default 50)
//   RR_TIMING_REPS       recorded repetitions per variant (default 3)
//

#include "gtest/gtest.h"
#include "rrConfig.h"
#include "rrRoadRunner.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace rr;

namespace {

    using Clock = std::chrono::steady_clock;

    int envInt(const char *name, int fallback) {
        const char *v = std::getenv(name);
        return v ? std::atoi(v) : fallback;
    }

    double elapsedMs(Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    }

    double median(std::vector<double> v) {
        std::sort(v.begin(), v.end());
        size_t n = v.size();
        return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    }

    // First order mass action network. Reaction i converts S(i % nSpecies) into
    // S((i * 7 + 1) % nSpecies), with its own rate constant. k0 is the first rate
    // constant and is varied between runs so that every model is unique.
    std::string makeSBML(int nSpecies, int nReactions, double k0) {
        std::ostringstream s;
        s << std::setprecision(17);
        s << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
             "<sbml xmlns=\"http://www.sbml.org/sbml/level3/version1/core\" level=\"3\" version=\"1\">\n"
             "<model id=\"timing\">\n"
             "<listOfCompartments>\n"
             "<compartment id=\"c\" spatialDimensions=\"3\" size=\"1\" constant=\"true\"/>\n"
             "</listOfCompartments>\n"
             "<listOfSpecies>\n";
        for (int i = 0; i < nSpecies; ++i) {
            s << "<species id=\"S" << i << "\" compartment=\"c\" initialConcentration=\"1\""
              << " hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/>\n";
        }
        s << "</listOfSpecies>\n<listOfParameters>\n";
        for (int i = 0; i < nReactions; ++i) {
            s << "<parameter id=\"k" << i << "\" value=\"" << (i == 0 ? k0 : 0.001) << "\" constant=\"true\"/>\n";
        }
        s << "</listOfParameters>\n<listOfReactions>\n";
        for (int i = 0; i < nReactions; ++i) {
            int from = i % nSpecies;
            int to = (i * 7 + 1) % nSpecies;
            if (to == from) {
                to = (to + 1) % nSpecies;
            }
            s << "<reaction id=\"R" << i << "\" reversible=\"false\" fast=\"false\">\n"
              << "<listOfReactants><speciesReference species=\"S" << from
              << "\" stoichiometry=\"1\" constant=\"true\"/></listOfReactants>\n"
              << "<listOfProducts><speciesReference species=\"S" << to
              << "\" stoichiometry=\"1\" constant=\"true\"/></listOfProducts>\n"
              << "<kineticLaw><math xmlns=\"http://www.w3.org/1998/Math/MathML\"><apply><times/>"
              << "<ci>k" << i << "</ci><ci>S" << from << "</ci></apply></math></kineticLaw>\n"
              << "</reaction>\n";
        }
        s << "</listOfReactions>\n</model>\n</sbml>\n";
        return s.str();
    }

    struct Variant {
        std::string label;
        int backend;
        int threads;
    };
}

TEST(LLJitCompileThreadsTimingTests, LoadTimeVersusNumThreads) {
    const int nReactions = envInt("RR_TIMING_REACTIONS", 2000);
    const int nSpecies = envInt("RR_TIMING_SPECIES", 500);
    const int reps = envInt("RR_TIMING_REPS", 3);
    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) {
        hw = 1;
    }

    std::vector<int> threadCounts = {0, 1, 2, 4, (int) hw};
    std::sort(threadCounts.begin(), threadCounts.end());
    threadCounts.erase(std::unique(threadCounts.begin(), threadCounts.end()), threadCounts.end());

    std::vector<Variant> variants;
    for (int n : threadCounts) {
        variants.push_back({"LLJit, " + std::to_string(n) + " thread(s)", Config::LLJIT, n});
    }
    variants.push_back({"MCJit (reference)", Config::MCJIT, 1});

    std::vector<std::vector<double>> loadMs(variants.size());

    int counter = 0;
    int done = 0;
    const int totalRuns = (int) variants.size() * (reps + 1);
    std::cout << std::fixed << std::setprecision(1);
    // rep -1 is a warm-up and is not recorded. The LLJit variants rotate their starting point each rep
    // so no LLJit variant is always first.
    for (int rep = -1; rep < reps; ++rep) {
        for (size_t k = 0; k < variants.size(); ++k) {
            // MCJit (the last variant) always goes first; the others rotate after it.
            size_t v = k == 0 ? variants.size() - 1 : (k + rep) % (variants.size() - 1);
            Config::resetToDefaults();
            Config::setValue(Config::LLVM_BACKEND, variants[v].backend);
            Config::setValue(Config::LLJIT_NUM_THREADS, variants[v].threads);
            Config::setValue(Config::LOADSBMLOPTIONS_RECOMPILE, true);
            Config::setValue(Config::LLVM_SYMBOL_CACHE, false);

            std::string sbml = makeSBML(nSpecies, nReactions, 0.001 + 1e-6 * ++counter);

            Clock::time_point t0 = Clock::now();
            RoadRunner rr(sbml);
            Clock::time_point t1 = Clock::now();
            if (rep >= 0) {
                loadMs[v].push_back(elapsedMs(t0, t1));
            }
            std::cout << "[" << ++done << "/" << totalRuns << "] " << variants[v].label
                      << (rep < 0 ? " (warm-up)" : "") << ": " << elapsedMs(t0, t1) << " ms" << std::endl;
        }
    }
    Config::resetToDefaults();

    std::cout << "model: " << nSpecies << " species, " << nReactions << " reactions; "
              << reps << " reps; " << hw << " logical cores\n"
              << "load = RoadRunner constructor (parse + codegen + JIT compile). Times in ms.\n"
              << std::left << std::setw(24) << "variant" << std::right
              << std::setw(12) << "load min" << std::setw(12) << "load med" << "\n";
    std::cout << std::fixed << std::setprecision(1);
    for (size_t v = 0; v < variants.size(); ++v) {
        std::cout << std::left << std::setw(24) << variants[v].label << std::right
                  << std::setw(12) << *std::min_element(loadMs[v].begin(), loadMs[v].end())
                  << std::setw(12) << median(loadMs[v]) << "\n";
        ASSERT_GT(median(loadMs[v]), 0.0);
    }
}
