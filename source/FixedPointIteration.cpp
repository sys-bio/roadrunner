//
// Created by Ciaran on 26/02/2021.
//

#include "FixedPointIteration.h"
#include "KinsolErrHandler.h"

namespace rr {

    FixedPointIteration::FixedPointIteration(ExecutableModel *executableModel)
            : KinsolSteadyStateSolver(executableModel) {
        // note: we deliberately use the NewtonIteration namespace here
        // because calling virtual methods from constructors is dangerous.
        // We *must* ensure we call the right version of createKinsol
        FixedPointIteration::resetSettings();
        FixedPointIteration::createKinsol();
    }

    FixedPointIteration::~FixedPointIteration() noexcept {
        FixedPointIteration::freeKinsol();
    }

    void FixedPointIteration::createKinsol() {
        int err;

        // call shared initialization protocol
        KinsolSteadyStateSolver::createKinsol();

        int stateVectorSize = mModel->getStateVector(nullptr);

        // kinsolDyDtFcn has the signature of "KINSysFn"
        if ((err = KINInit(mKinsol_Memory, kinsolDyDtFcn < FixedPointIteration > , mStateVector)) != KIN_SUCCESS) {
            decodeKinsolError(err);
        }
    }

    void FixedPointIteration::freeKinsol() {
        KinsolSteadyStateSolver::freeKinsol();
    }

    double FixedPointIteration::solve() {
        return solveForSteadyState(this, KIN_FP);
    }

    void FixedPointIteration::updateKinsol() {
        KinsolSteadyStateSolver::updateKinsol();
        KINSetDampingAA(mKinsol_Memory, (double)getValue("damping_aa"));
        KINSetMAA(mKinsol_Memory, (long)getValue("maa"));
    }

    void FixedPointIteration::resetSettings() {
        KinsolSteadyStateSolver::resetSettings();

        std::string desc = "Anderson Acceleration subspace size. Default is 0, no acceleration.";
        addSetting("maa", Setting(0), "Anderson Acceleration", desc, desc);

        desc = "Anderson Acceleration damping parameter";
        addSetting("damping_aa", Setting(1.0), "Anderson Acceleration Damping Parameter. Default 1=no damping", desc, desc);

    }

    std::string FixedPointIteration::getName() const {
        return "FixedPointIteration";
    }

    std::string FixedPointIteration::getDescription() const {
        return "Uses the sundials implementation of fixed point "
               "iteration";
    }

    std::string FixedPointIteration::getHint() const {
        return "Fixed point iteration";
    }

}
