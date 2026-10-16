// Regression tests for the LLJIT section memory (sys-bio/roadrunner issue 1368).
//
// The code generated for a model holds 32-bit offsets between the code section and the
// read-only data section (jump tables, PC-relative references). LLJit therefore allocates all
// sections of one object from a single contiguous range. Without that, a large model plus enough
// address space in use elsewhere in the process could put the sections more than 2 GiB apart,
// and the model crashed when it was loaded or simulated.
//
// The tests are portable. SectionsStayWithinTwoGiB checks the invariant directly on every
// platform: LLJit measures how far apart the sections of each object ended up, and the test asks
// for read-only data far from the code (RR_LLJIT_FAR_RODATA), which only a missing range allows.
// With the range turned off (RR_LLJIT_RESERVE=0) it fails everywhere the far placement works.
// The other tests reproduce the original crash on Linux with the range turned off, and elsewhere
// check that loading, simulating and releasing models through the range works.

#include "gtest/gtest.h"
#include "llvm/LLJit.h"
#include "rrConfig.h"
#include "rrRoadRunner.h"
#include "rrRoadRunnerOptions.h"

#include <cmath>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#ifndef MAP_NORESERVE
#define MAP_NORESERVE 0
#endif
#endif

using namespace rr;

// the generated code holds signed 32-bit offsets between an object's sections
const std::uint64_t twoGiB = std::uint64_t(1) << 31;

namespace {

    const double stopTime = 5.0;

    double rateConstant(int i) {
        return 0.05 + 0.01 * (i % 7);
    }

    // Decay model: n species S<i> with rate constants k<i>, so that S<i>(t) = exp(-k<i> * t).
    // The generated code grows with n.
    std::string decayModel(int n) {
        std::ostringstream sbml;
        sbml << R"(<?xml version="1.0" encoding="UTF-8"?>)"
             << R"(<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">)"
             << R"(<model id="decay"><listOfCompartments><compartment id="c" size="1" constant="true"/>)"
             << R"(</listOfCompartments><listOfSpecies>)";
        for (int i = 0; i < n; ++i) {
            sbml << R"(<species id="S)" << i << R"(" compartment="c" initialConcentration="1")"
                 << R"( hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>)";
        }
        sbml << "</listOfSpecies><listOfParameters>";
        for (int i = 0; i < n; ++i) {
            sbml << R"(<parameter id="k)" << i << R"(" value=")" << rateConstant(i) << R"(" constant="true"/>)";
        }
        sbml << "</listOfParameters><listOfReactions>";
        for (int i = 0; i < n; ++i) {
            sbml << R"(<reaction id="R)" << i << R"(" reversible="false"><listOfReactants>)"
                 << R"(<speciesReference species="S)" << i << R"(" stoichiometry="1" constant="true"/>)"
                 << R"(</listOfReactants><kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">)"
                 << R"(<apply><times/><ci>k)" << i << R"(</ci><ci>S)" << i << R"(</ci></apply>)"
                 << R"(</math></kineticLaw></reaction>)";
        }
        sbml << "</listOfReactions></model></sbml>";
        return sbml.str();
    }

    // Address space that is reserved but never touched, like the reporter's script: it moves
    // later mappings apart, so a model's sections do not end up next to each other by luck.
    class AddressSpaceReservation {
    public:
        explicit AddressSpaceReservation(size_t bytes) : bytes(bytes) {
#ifdef _WIN32
            base = VirtualAlloc(nullptr, bytes, MEM_RESERVE, PAGE_NOACCESS);
#else
            void *p = mmap(nullptr, bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
            base = p == MAP_FAILED ? nullptr : p;
#endif
        }

        ~AddressSpaceReservation() {
            if (base != nullptr) {
#ifdef _WIN32
                VirtualFree(base, 0, MEM_RELEASE);
#else
                munmap(base, bytes);
#endif
            }
        }

        AddressSpaceReservation(const AddressSpaceReservation &) = delete;

        AddressSpaceReservation &operator=(const AddressSpaceReservation &) = delete;

        bool ok() const { return base != nullptr; }

    private:
        size_t bytes;
        void *base = nullptr;
    };

    void setEnv(const char *name, const char *value) {
#ifdef _WIN32
        _putenv_s(name, value);
#else
        setenv(name, value, 1);
#endif
    }

    void clearEnv(const char *name) {
#ifdef _WIN32
        _putenv_s(name, "");
#else
        unsetenv(name);
#endif
    }

    std::unique_ptr<RoadRunner> loadWithLLJit(const std::string &sbml) {
        Config::setValue(Config::LLVM_BACKEND, Config::LLVM_BACKEND_VALUES::LLJIT);
        LoadSBMLOptions opt;
        opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::RECOMPILE;
        opt.setLLJitOptimizationLevel(LoadSBMLOptions::LLJIT_OPTIMIZATION_LEVELS::NONE);
        return std::make_unique<RoadRunner>(sbml, &opt);
    }

    // Simulates and compares a spread of species with the analytic solution.
    void simulateAndCheck(RoadRunner &r, int n) {
        r.simulate(0, stopTime, 11);
        for (int i : {0, 1, n / 3, n / 2, n - 2, n - 1}) {
            EXPECT_NEAR(r.getValue("S" + std::to_string(i)), std::exp(-rateConstant(i) * stopTime), 1e-4)
                            << "species S" << i;
        }
    }

}

// A model large enough to need a big code section loads, simulates and is released.
TEST(LLJitSectionRange, LargeModel) {
    const int n = 2000;
    auto r = loadWithLLJit(decayModel(n));
    simulateAndCheck(*r, n);
}

// The original report: with address space reserved around it, a large model's code and
// read-only data used to end up more than 2 GiB apart.
TEST(LLJitSectionRange, LargeModelWithAddressSpaceReserved) {
    if (sizeof(void *) < 8) {
        GTEST_SKIP() << "needs a 64-bit address space";
    }
    AddressSpaceReservation reservation(size_t(3) << 30);
    if (!reservation.ok()) {
        GTEST_SKIP() << "could not reserve 3 GiB of address space";
    }
    rrllvm::LLJit::largestSectionSpan = 0;
    const int n = 2000;
    auto r = loadWithLLJit(decayModel(n));
    EXPECT_LT(rrllvm::LLJit::largestSectionSpan.load(), twoGiB);
    simulateAndCheck(*r, n);
}

// The invariant itself, independent of how the address space happens to be laid out: even when
// read-only data is requested far from the code, the sections of an object stay together.
TEST(LLJitSectionRange, SectionsStayWithinTwoGiB) {
#if !defined(__x86_64__) && !defined(_M_X64)
    GTEST_SKIP() << "the section range is only used on x86-64";
#endif
    setEnv("RR_LLJIT_FAR_RODATA", "1");
    rrllvm::LLJit::largestSectionSpan = 0;
    const int n = 300;
    auto r = loadWithLLJit(decayModel(n));
    const std::uint64_t span = rrllvm::LLJit::largestSectionSpan.load();
    clearEnv("RR_LLJIT_FAR_RODATA");
    EXPECT_GT(span, std::uint64_t(0)) << "the section span was not measured";
    EXPECT_LT(span, twoGiB);
    simulateAndCheck(*r, n);
}

// The original report's trigger: a second model compiled while the first is still alive.
TEST(LLJitSectionRange, TwoModelsAliveTogether) {
    const int n = 400;
    auto first = loadWithLLJit(decayModel(n));
    auto second = loadWithLLJit(decayModel(n + 100));
    simulateAndCheck(*second, n + 100);
    simulateAndCheck(*first, n);
}

// Each model's range is released with the model; loading and dropping repeatedly keeps working.
TEST(LLJitSectionRange, RepeatedLoadAndRelease) {
    const int n = 300;
    for (int round = 0; round < 5; ++round) {
        auto r = loadWithLLJit(decayModel(n));
        simulateAndCheck(*r, n);
    }
}

// The switch that turns the range off keeps the plain allocator usable.
TEST(LLJitSectionRange, RangeDisabledStillWorks) {
    setEnv("RR_LLJIT_RESERVE", "0");
    const int n = 300;
    auto r = loadWithLLJit(decayModel(n));
    clearEnv("RR_LLJIT_RESERVE");
    simulateAndCheck(*r, n);
}
