// Time of one Jacobian of a model: the Enzyme Jacobian against the difference quotient of CVODE.
//
// usage: enzyme_jacobian_timing <sbml> <time> [<parameter> <value>] [reps]
//
// Loads the model with the Enzyme Jacobian, sets the parameter (a dose), simulates to <time> and
// times, at that state, one evaluation of the state vector rate and one Enzyme Jacobian (CSC
// values, as the CVODE callback gets them), as medians over <reps> repetitions of a batch. The
// dense difference quotient of CVODE costs one state vector rate per state, so its time is
// estimated as n times the time of one rate. Also reports the time of loading the model with the
// Enzyme Jacobian (compiled anew). Prints one JSON object.
//
// The JIT backend comes from the environment, as in models.py: RR_LLVM_BACKEND (mcjit, the
// default, or lljit) and RR_LLJIT_OPT (none, the default, less, default or aggressive).

#include "rrRoadRunner.h"
#include "rrConfig.h"
#include "rrExecutableModel.h"
#include "rrLogger.h"
#include "llvm/JacobianSparsity.h"
#include "llvm/LLVMModelDataSymbols.h"

#include <sbml/SBMLTypes.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
    void applyBackendFromEnvironment() {
        const char *backend = std::getenv("RR_LLVM_BACKEND");
        if (backend == nullptr || std::string(backend) == "mcjit") {
            rr::Config::setValue(rr::Config::LLVM_BACKEND, rr::Config::MCJIT);
            return;
        }
        if (std::string(backend) != "lljit") {
            throw std::invalid_argument("RR_LLVM_BACKEND must be mcjit or lljit");
        }
        rr::Config::setValue(rr::Config::LLVM_BACKEND, rr::Config::LLJIT);
        const char *opt = std::getenv("RR_LLJIT_OPT");
        const std::vector<std::string> levels{"none", "less", "default", "aggressive"};
        auto it = std::find(levels.begin(), levels.end(), opt == nullptr ? "none" : opt);
        if (it == levels.end()) {
            throw std::invalid_argument("RR_LLJIT_OPT must be none, less, default or aggressive");
        }
        rr::Config::setValue(rr::Config::LLJIT_OPTIMIZATION_LEVEL, static_cast<int>(it - levels.begin()));
    }

    template<typename F>
    double medianSeconds(F f, int reps, int batch) {
        std::vector<double> t;
        for (int r = 0; r < reps; ++r) {
            auto start = std::chrono::steady_clock::now();
            for (int b = 0; b < batch; ++b) {
                f();
            }
            auto end = std::chrono::steady_clock::now();
            t.push_back(std::chrono::duration<double>(end - start).count() / batch);
        }
        std::sort(t.begin(), t.end());
        return t[t.size() / 2];
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "usage: enzyme_jacobian_timing <sbml> <time> [<parameter> <value>] [reps]" << std::endl;
        return 2;
    }
    std::string sbml = argv[1];
    double time = std::stod(argv[2]);
    std::string parameter = argc >= 5 ? argv[3] : "";
    double value = argc >= 5 ? std::stod(argv[4]) : 0.0;
    int reps = argc >= 6 ? std::stoi(argv[5]) : (argc == 4 ? std::stoi(argv[3]) : 21);

    rr::Logger::setLevel(rr::Logger::LOG_ERROR);
    applyBackendFromEnvironment();
    rr::Config::setValue(rr::Config::LLVM_ENZYME_JACOBIAN, true);
    rr::Config::setValue(rr::Config::LOADSBMLOPTIONS_RECOMPILE, true);
    auto loadStart = std::chrono::steady_clock::now();
    rr::RoadRunner r(sbml);
    double tLoad = std::chrono::duration<double>(std::chrono::steady_clock::now() - loadStart).count();
    if (!parameter.empty()) {
        r.setValue(parameter, value);
    }
    if (time > 0) {
        r.simulate(0, time, 2);
    }
    rr::ExecutableModel *model = r.getModel();
    if (!model->hasStateVectorJacobian()) {
        std::cerr << "no Enzyme Jacobian: " << model->getStateVectorJacobianUnavailableReason() << std::endl;
        return 1;
    }
    const rr::CscPattern &pattern = *model->getStateVectorJacobianPattern();
    int n = model->getStateVector(nullptr);
    std::vector<double> y(n), dydt(n), values(pattern.nnz());
    model->getStateVector(y.data());
    double t = model->getTime();

    // the coloring is not part of the model API; recompute it from the SBML as roadrunner does
    std::unique_ptr<libsbml::SBMLDocument> doc(libsbml::readSBMLFromString(r.getSBML().c_str()));
    rrllvm::LLVMModelDataSymbols symbols(doc->getModel(), 0);
    rrllvm::JacobianSparsity sparsity(*doc->getModel(), symbols);

    int batch = std::max(1, 20000 / std::max(n, 1));
    double tRate = medianSeconds([&]() { model->getStateVectorRate(t, y.data(), dydt.data()); }, reps, batch * 10);
    double tEnzyme = medianSeconds([&]() { model->getStateVectorJacobian(t, y.data(), values.data()); },
                                   reps, std::max(1, batch / 4));

    std::cout << "{\"states\": " << n << ", \"nnz\": " << pattern.nnz() << ", \"colors\": " << sparsity.numColors()
              << ", \"t_load\": " << tLoad << ", \"t_rate\": " << tRate << ", \"t_dq\": " << n * tRate
              << ", \"t_enzyme\": " << tEnzyme << "}" << std::endl;
    return 0;
}
