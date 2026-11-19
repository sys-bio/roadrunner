// Regression tests for the LLJIT section memory (sys-bio/roadrunner issue 1368).
//
// The code generated for a model holds 32-bit offsets between the code section and the
// read-only data section (jump tables, PC-relative references). LLJit therefore allocates all
// sections of one object from a single contiguous range. Without that, the sections of a model
// could end up more than 2 GiB apart, and the model crashed when it was loaded or simulated.
//
// The tests ask LLJit to place any section that does not come from the contiguous range about
// 3 GiB from the code, load a model, and check how far apart LLJit put the sections of the
// object. With the range they stay together on every platform. Without it they are 3 GiB
// apart: the check fails where the model still loads (Windows, x86_64 macOS), and the process
// crashes where the 32-bit offsets are used (Linux, arm64 macOS).

#include "gtest/gtest.h"
#include "llvm/LLJitSectionMemory.h"
#include "rrConfig.h"
#include "rrRoadRunner.h"
#include "rrRoadRunnerOptions.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>

using namespace rr;
using rrllvm::LLJitSectionMemoryTesting;

namespace {

    // the generated code holds signed 32-bit offsets between an object's sections
    const std::uint64_t twoGiB = std::uint64_t(1) << 31;

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

class LLJitSectionRange : public ::testing::Test {
protected:
    void SetUp() override {
        if (sizeof(void *) < 8) {
            GTEST_SKIP() << "needs a 64-bit address space";
        }
        LLJitSectionMemoryTesting::farDataGiB = 3;
        LLJitSectionMemoryTesting::largestSectionSpan = 0;
    }

    void TearDown() override {
        LLJitSectionMemoryTesting::farDataGiB = 0;
    }

    // Loads a model and checks that the sections of its object stayed together.
    std::unique_ptr<RoadRunner> load(int n) {
        auto r = loadWithLLJit(decayModel(n));
        const std::uint64_t span = LLJitSectionMemoryTesting::largestSectionSpan.load();
        EXPECT_GT(span, std::uint64_t(0)) << "the section span was not measured";
        EXPECT_LT(span, twoGiB) << "the sections of a model are " << span << " bytes apart";
        return r;
    }
};

// The sections of a model stay together, and the model works.
TEST_F(LLJitSectionRange, SectionsStayWithinTwoGiB) {
    const int n = 300;
    auto r = load(n);
    simulateAndCheck(*r, n);
}

// The original report's trigger: a second model compiled while the first is still alive.
TEST_F(LLJitSectionRange, TwoModelsAliveTogether) {
    const int n = 300;
    auto first = load(n);
    auto second = load(n + 100);
    simulateAndCheck(*second, n + 100);
    simulateAndCheck(*first, n);
}

// Prints the stack of the first access violations in this process.
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <dbghelp.h>
#include <atomic>
#include <cstdint>
#include <cstdio>

#pragma comment(lib, "dbghelp.lib")

namespace {

    LONG CALLBACK printAccessViolationStack(EXCEPTION_POINTERS *info) {
        static std::atomic<int> printed{0};
        const EXCEPTION_RECORD *record = info->ExceptionRecord;
        if (record->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || printed++ >= 3) {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        const HANDLE process = GetCurrentProcess();
        const HANDLE thread = GetCurrentThread();
        SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
        SymInitialize(process, nullptr, TRUE);

        const ULONG_PTR kind = record->ExceptionInformation[0];
        std::fprintf(stderr, "\n=== ACCESS VIOLATION: %s of address %p, instruction at %p ===\n",
                     kind == 0 ? "read" : kind == 1 ? "write" : "execute",
                     reinterpret_cast<void *>(record->ExceptionInformation[1]), record->ExceptionAddress);

        CONTEXT context = *info->ContextRecord;
        STACKFRAME64 frame = {};
        frame.AddrPC.Offset = context.Rip;
        frame.AddrPC.Mode = AddrModeFlat;
        frame.AddrFrame.Offset = context.Rbp;
        frame.AddrFrame.Mode = AddrModeFlat;
        frame.AddrStack.Offset = context.Rsp;
        frame.AddrStack.Mode = AddrModeFlat;

        for (int i = 0; i < 64; ++i) {
            if (!StackWalk64(IMAGE_FILE_MACHINE_AMD64, process, thread, &frame, &context, nullptr,
                             SymFunctionTableAccess64, SymGetModuleBase64, nullptr) ||
                frame.AddrPC.Offset == 0) {
                break;
            }
            const DWORD64 pc = frame.AddrPC.Offset;

            char module[MAX_PATH] = "?";
            HMODULE handle = nullptr;
            if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                   reinterpret_cast<LPCSTR>(static_cast<std::uintptr_t>(pc)), &handle)) {
                GetModuleFileNameA(handle, module, MAX_PATH);
            }

            alignas(SYMBOL_INFO) char buffer[sizeof(SYMBOL_INFO) + 512];
            SYMBOL_INFO *symbol = reinterpret_cast<SYMBOL_INFO *>(buffer);
            symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
            symbol->MaxNameLen = 512;
            DWORD64 symbolOffset = 0;
            const bool haveSymbol = SymFromAddr(process, pc, &symbolOffset, symbol) != FALSE;

            IMAGEHLP_LINE64 line = {};
            line.SizeOfStruct = sizeof(line);
            DWORD lineOffset = 0;
            const bool haveLine = SymGetLineFromAddr64(process, pc, &lineOffset, &line) != FALSE;

            std::fprintf(stderr, "  #%d %p %s+0x%llx  %s:%lu  [%s]\n", i, reinterpret_cast<void *>(pc),
                         haveSymbol ? symbol->Name : "(no symbol)",
                         static_cast<unsigned long long>(symbolOffset),
                         haveLine ? line.FileName : "(no line)", haveLine ? line.LineNumber : 0UL, module);
        }
        std::fflush(stderr);
        return EXCEPTION_CONTINUE_SEARCH;
    }

    struct InstallStackPrinter {
        InstallStackPrinter() {
            AddVectoredExceptionHandler(1, printAccessViolationStack);
        }
    } installStackPrinter;
}
#endif
