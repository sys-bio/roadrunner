//
// Crash hunt for the sporadic LLJit segfault in sys-bio/roadrunner#1368.
//
// Each run is a fresh process (gtest threadsafe death test, which re-executes
// the test binary, so every run gets new ASLR). The child repeatedly loads a
// model with LLJIT + RECOMPILE + no optimization, keeping the previous runner
// alive while the next one is built. How each child ended is recorded and
// summarised per arm. A crashing child writes a report with the fault PC,
// fault address, stack words and the memory map (/proc/self/maps on Linux,
// VirtualQuery regions on Windows).
//
// Environment variables:
//   RR_CRASHHUNT_SBML           path to the SBML model (default: synthetic model)
//   RR_CRASHHUNT_RUNS           processes per arm (default 10)
//   RR_CRASHHUNT_LOADS          model loads per process (default 3)
//   RR_CRASHHUNT_SPECIES        species in the synthetic model (default 300)
//   RR_CRASHHUNT_ARMS           comma separated substrings selecting arms (default all)
//   RR_CRASHHUNT_RELEASE_FIRST  1: free the previous runner before the next load
//   RR_CRASHHUNT_SIMULATE       1: run a short simulation after each load
//   RR_CRASHHUNT_DIR            directory for progress and crash reports
//   RR_CRASHHUNT_FAIL_ON_CRASH  1: fail the test if any run crashes
//
// Run with: --gtest_filter=LLJitCrashHunt*
//
// The InProcess* tests run one arm once in the test process itself, with no crash handlers,
// and skip themselves unless a debugger is attached (Test Explorer > Debug on one test).
//

#include "gtest/gtest.h"
#include "rrRoadRunner.h"
#include "rrRoadRunnerOptions.h"
#include "rrConfig.h"
#include "sbml/SBMLTypes.h"
#include "sbml/math/L3Parser.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/personality.h>
#include <sys/wait.h>
#include <ucontext.h>
#include <unistd.h>
#define CRASHHUNT_SUPPORTED 1
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <csignal>
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#define setenv(name, value, overwrite) _putenv_s(name, value)
#define unsetenv(name) _putenv_s(name, "")
#define CRASHHUNT_SUPPORTED 1
#endif

using namespace rr;

namespace {

    int envInt(const char *name, int dflt) {
        const char *v = std::getenv(name);
        return v ? std::atoi(v) : dflt;
    }

    std::string envStr(const char *name, const std::string &dflt) {
        const char *v = std::getenv(name);
        return v ? std::string(v) : dflt;
    }

    // ---------------------------------------------------------------- model

    void addMath(const std::string &formula, std::function<void(const libsbml::ASTNode *)> setter) {
        libsbml::ASTNode *ast = libsbml::SBML_parseL3Formula(formula.c_str());
        if (!ast) {
            throw std::runtime_error("could not parse formula: " + formula);
        }
        setter(ast);
        delete ast;
    }

    // A chain-like network with mass-action and Michaelis-Menten steps, a piecewise
    // dosing input, initial assignments, assignment rules and a few events.
    std::string buildSyntheticSBML(int numSpecies) {
        const int numCompartments = 8;
        libsbml::SBMLDocument doc(3, 2);
        libsbml::Model *model = doc.createModel("synthetic");

        for (int c = 0; c < numCompartments; ++c) {
            libsbml::Compartment *comp = model->createCompartment();
            comp->setId("c" + std::to_string(c));
            comp->setConstant(true);
            comp->setSize(1.0 + 0.1 * c);
            comp->setSpatialDimensions(3.0);
        }

        auto addParam = [&](const std::string &id, double value, bool constant) {
            libsbml::Parameter *p = model->createParameter();
            p->setId(id);
            p->setValue(value);
            p->setConstant(constant);
        };
        addParam("bw", 75.0, true);
        addParam("dose", 1.5, false);
        addParam("tDose", 24.0, true);

        for (int i = 0; i < numSpecies; ++i) {
            libsbml::Species *s = model->createSpecies();
            s->setId("S" + std::to_string(i));
            s->setCompartment("c" + std::to_string(i % numCompartments));
            s->setInitialConcentration(0.0);
            s->setBoundaryCondition(false);
            s->setHasOnlySubstanceUnits(false);
            s->setConstant(false);
            addParam("frac" + std::to_string(i), 0.001 * (1 + i % 17), true);
            addParam("k" + std::to_string(i), 0.01 + 0.0005 * (i % 23), true);

            libsbml::InitialAssignment *ia = model->createInitialAssignment();
            ia->setSymbol(s->getId());
            addMath("bw * frac" + std::to_string(i) + " * 0.1",
                    [&](const libsbml::ASTNode *ast) { ia->setMath(ast); });
        }

        for (int i = 0; i + 1 < numSpecies; ++i) {
            libsbml::Reaction *r = model->createReaction();
            r->setId("R" + std::to_string(i));
            r->setReversible(false);
            r->setFast(false);
            libsbml::SpeciesReference *rct = r->createReactant();
            rct->setSpecies("S" + std::to_string(i));
            rct->setStoichiometry(1.0);
            rct->setConstant(true);
            libsbml::SpeciesReference *prd = r->createProduct();
            prd->setSpecies("S" + std::to_string(i + 1));
            prd->setStoichiometry(1.0);
            prd->setConstant(true);
            std::string comp = "c" + std::to_string(i % numCompartments);
            std::string law = (i % 3 == 0)
                              ? "k" + std::to_string(i) + " * S" + std::to_string(i) + " * " + comp
                              : "k" + std::to_string(i) + " * S" + std::to_string(i) + " / (0.5 + S" +
                                std::to_string(i) + ") * " + comp;
            libsbml::KineticLaw *kl = r->createKineticLaw();
            addMath(law, [&](const libsbml::ASTNode *ast) { kl->setMath(ast); });
        }

        libsbml::Reaction *dosing = model->createReaction();
        dosing->setId("Rdose");
        dosing->setReversible(false);
        dosing->setFast(false);
        libsbml::SpeciesReference *dosed = dosing->createProduct();
        dosed->setSpecies("S0");
        dosed->setStoichiometry(1.0);
        dosed->setConstant(true);
        libsbml::KineticLaw *dosingLaw = dosing->createKineticLaw();
        addMath("piecewise(dose, time < tDose, 0) * c0",
                [&](const libsbml::ASTNode *ast) { dosingLaw->setMath(ast); });

        int numDerived = numSpecies / 2;
        for (int j = 0; j < numDerived; ++j) {
            addParam("a" + std::to_string(j), 0.0, false);
            libsbml::AssignmentRule *ar = model->createAssignmentRule();
            ar->setVariable("a" + std::to_string(j));
            std::string prev = (j % 5 == 0) ? "bw * 0.75" : "a" + std::to_string(j - 1) + " * 1.0001";
            addMath(prev + " + S" + std::to_string(j) + " / c" + std::to_string(j % numCompartments),
                    [&](const libsbml::ASTNode *ast) { ar->setMath(ast); });
        }

        for (int e = 0; e < 3; ++e) {
            libsbml::Event *ev = model->createEvent();
            ev->setId("E" + std::to_string(e));
            ev->setUseValuesFromTriggerTime(true);
            libsbml::Trigger *trig = ev->createTrigger();
            trig->setInitialValue(false);
            trig->setPersistent(true);
            addMath("time > " + std::to_string(48 * (e + 1)),
                    [&](const libsbml::ASTNode *ast) { trig->setMath(ast); });
            libsbml::EventAssignment *ea = ev->createEventAssignment();
            ea->setVariable("dose");
            addMath("dose * 1.5", [&](const libsbml::ASTNode *ast) { ea->setMath(ast); });
        }

        char *text = libsbml::writeSBMLToString(&doc);
        std::string result(text);
        free(text);
        return result;
    }

    std::string loadModelText() {
        std::string path = envStr("RR_CRASHHUNT_SBML", "");
        if (path.empty()) {
            return buildSyntheticSBML(envInt("RR_CRASHHUNT_SPECIES", 300));
        }
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("cannot read RR_CRASHHUNT_SBML: " + path);
        }
        std::stringstream ss;
        ss << in.rdbuf();
        return ss.str();
    }

    // ---------------------------------------------------------- statistics

    double logChoose(int n, int k) {
        return std::lgamma(n + 1.0) - std::lgamma(k + 1.0) - std::lgamma(n - k + 1.0);
    }

    double binomCdf(int k, int n, double p) {
        double sum = 0;
        for (int i = 0; i <= k; ++i) {
            sum += std::exp(logChoose(n, i) + i * std::log(p) + (n - i) * std::log1p(-p));
        }
        return sum;
    }

    // One-sided 95% upper bound on the crash probability given k crashes in n runs.
    double upperBound95(int k, int n) {
        if (n == 0 || k >= n) {
            return 1.0;
        }
        double lo = 0.0, hi = 1.0;
        for (int it = 0; it < 100; ++it) {
            double mid = 0.5 * (lo + hi);
            if (binomCdf(k, n, mid) > 0.05) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return hi;
    }

    // Two-sided Fisher exact test on [[a, b], [c, d]].
    double fisherExact(int a, int b, int c, int d) {
        int r1 = a + b, r2 = c + d, c1 = a + c, total = r1 + r2;
        if (total == 0) {
            return 1.0;
        }
        auto logP = [&](int x) {
            return logChoose(r1, x) + logChoose(r2, c1 - x) - logChoose(total, c1);
        };
        double pObserved = std::exp(logP(a));
        double p = 0;
        for (int x = std::max(0, c1 - r2); x <= std::min(r1, c1); ++x) {
            double px = std::exp(logP(x));
            if (px <= pObserved * (1 + 1e-7)) {
                p += px;
            }
        }
        return std::min(1.0, p);
    }

#ifdef CRASHHUNT_SUPPORTED

    // ------------------------------------------------- child: crash report

    char gReportPath[1024];
    char gProgressPath[1024];

#ifdef _WIN32
    typedef HANDLE FileHandle;
#else
    typedef int FileHandle;
#endif

    void writeAll(FileHandle fd, const char *s, size_t n) {
        while (n > 0) {
#ifdef _WIN32
            DWORD w = 0;
            if (!WriteFile(fd, s, static_cast<DWORD>(n), &w, nullptr) || w == 0) {
                return;
            }
#else
            ssize_t w = ::write(fd, s, n);
            if (w <= 0) {
                return;
            }
#endif
            s += w;
            n -= static_cast<size_t>(w);
        }
    }

    void writeStr(FileHandle fd, const char *s) {
        writeAll(fd, s, std::strlen(s));
    }

    void writeHex(FileHandle fd, unsigned long long v) {
        char buf[18];
        buf[0] = '0';
        buf[1] = 'x';
        for (int i = 0; i < 16; ++i) {
            buf[2 + i] = "0123456789abcdef"[(v >> (60 - 4 * i)) & 0xf];
        }
        writeAll(fd, buf, sizeof(buf));
    }

    void writeDec(FileHandle fd, int v) {
        char buf[16];
        int n = 0;
        if (v == 0) {
            buf[n++] = '0';
        }
        char rev[16];
        int m = 0;
        while (v > 0 && m < 15) {
            rev[m++] = static_cast<char>('0' + v % 10);
            v /= 10;
        }
        while (m > 0) {
            buf[n++] = rev[--m];
        }
        writeAll(fd, buf, static_cast<size_t>(n));
    }

#ifdef __linux__
    void crashHandler(int sig, siginfo_t *info, void *ctx) {
        int fd = ::open(gReportPath, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) {
            return;
        }
        writeStr(fd, "SIG ");
        writeDec(fd, sig);
        writeStr(fd, "\nADDR ");
        writeHex(fd, reinterpret_cast<unsigned long long>(info->si_addr));
        writeStr(fd, "\n");
#if defined(__x86_64__)
        ucontext_t *uc = static_cast<ucontext_t *>(ctx);
        unsigned long long rip = static_cast<unsigned long long>(uc->uc_mcontext.gregs[REG_RIP]);
        unsigned long long rsp = static_cast<unsigned long long>(uc->uc_mcontext.gregs[REG_RSP]);
        writeStr(fd, "RIP ");
        writeHex(fd, rip);
        writeStr(fd, "\nRSP ");
        writeHex(fd, rsp);
        writeStr(fd, "\n");
#else
        (void) ctx;
#endif
        writeStr(fd, "MAPS\n");
        int maps = ::open("/proc/self/maps", O_RDONLY);
        if (maps >= 0) {
            char buf[4096];
            ssize_t n;
            while ((n = ::read(maps, buf, sizeof(buf))) > 0) {
                writeAll(fd, buf, static_cast<size_t>(n));
            }
            ::close(maps);
        }
#if defined(__x86_64__)
        writeStr(fd, "STACK\n");
        const unsigned long long *sp = reinterpret_cast<const unsigned long long *>(rsp);
        for (int i = 0; i < 64; ++i) {
            writeHex(fd, sp[i]);
            writeStr(fd, "\n");
        }
#endif
        ::close(fd);
    }

    // Reports on an alternate stack and then re-raises with the default action.
    void installCrashHandler() {
        static char altStack[1 << 16];
        stack_t ss;
        ss.ss_sp = altStack;
        ss.ss_size = sizeof(altStack);
        ss.ss_flags = 0;
        sigaltstack(&ss, nullptr);

        struct sigaction sa;
        std::memset(&sa, 0, sizeof(sa));
        sa.sa_sigaction = crashHandler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_RESETHAND | SA_NODEFER;
        for (int s : {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGABRT}) {
            sigaction(s, &sa, nullptr);
        }
    }

#else

    // Reports on fatal exceptions before any other handler (including gtest's SEH catch) sees them.
    LONG CALLBACK crashHandler(EXCEPTION_POINTERS *info) {
        const DWORD code = info->ExceptionRecord->ExceptionCode;
        switch (code) {
            case EXCEPTION_ACCESS_VIOLATION:
            case EXCEPTION_ILLEGAL_INSTRUCTION:
            case EXCEPTION_PRIV_INSTRUCTION:
            case EXCEPTION_IN_PAGE_ERROR:
            case EXCEPTION_INT_DIVIDE_BY_ZERO:
            case EXCEPTION_STACK_OVERFLOW:
                break;
            default:
                return EXCEPTION_CONTINUE_SEARCH;
        }
        static volatile LONG entered = 0;
        if (InterlockedExchange(&entered, 1) != 0) {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        HANDLE fd = CreateFileA(gReportPath, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (fd != INVALID_HANDLE_VALUE) {
            writeStr(fd, "EXCEPTION ");
            writeHex(fd, code);
            writeStr(fd, "\nADDR ");
            if (code == EXCEPTION_ACCESS_VIOLATION && info->ExceptionRecord->NumberParameters >= 2) {
                writeHex(fd, info->ExceptionRecord->ExceptionInformation[1]);
            } else {
                writeHex(fd, reinterpret_cast<ULONG_PTR>(info->ExceptionRecord->ExceptionAddress));
            }
            writeStr(fd, "\n");
#if defined(_M_X64)
            const unsigned long long rsp = info->ContextRecord->Rsp;
            writeStr(fd, "RIP ");
            writeHex(fd, info->ContextRecord->Rip);
            writeStr(fd, "\nRSP ");
            writeHex(fd, rsp);
            writeStr(fd, "\n");
#endif
            writeStr(fd, "MAPS\n");
            SYSTEM_INFO si;
            GetSystemInfo(&si);
            const uintptr_t maxAddr = reinterpret_cast<uintptr_t>(si.lpMaximumApplicationAddress);
            uintptr_t addr = reinterpret_cast<uintptr_t>(si.lpMinimumApplicationAddress);
            MEMORY_BASIC_INFORMATION mbi;
            while (addr < maxAddr && VirtualQuery(reinterpret_cast<LPCVOID>(addr), &mbi, sizeof(mbi)) == sizeof(mbi)) {
                const uintptr_t base = reinterpret_cast<uintptr_t>(mbi.BaseAddress);
                const uintptr_t end = base + mbi.RegionSize;
                if (mbi.State != MEM_FREE) {
                    const DWORD prot = mbi.State == MEM_COMMIT ? (mbi.Protect & 0xff) : 0;
                    const bool exec = prot == PAGE_EXECUTE || prot == PAGE_EXECUTE_READ ||
                                      prot == PAGE_EXECUTE_READWRITE || prot == PAGE_EXECUTE_WRITECOPY;
                    const bool write = prot == PAGE_READWRITE || prot == PAGE_WRITECOPY ||
                                       prot == PAGE_EXECUTE_READWRITE || prot == PAGE_EXECUTE_WRITECOPY;
                    const bool read = prot != 0 && prot != PAGE_NOACCESS && prot != PAGE_EXECUTE;
                    char perms[6] = {read ? 'r' : '-', write ? 'w' : '-', exec ? 'x' : '-', 'p', ' ', 0};
                    writeHex(fd, base);
                    writeStr(fd, "-");
                    writeHex(fd, end);
                    writeStr(fd, " ");
                    writeStr(fd, perms);
                    writeStr(fd, mbi.State == MEM_COMMIT ? "commit " : "reserve ");
                    writeStr(fd, mbi.Type == MEM_IMAGE ? "image\n" : mbi.Type == MEM_MAPPED ? "mapped\n" : "private\n");
                }
                if (end <= addr) {
                    break;
                }
                addr = end;
            }
#if defined(_M_X64)
            writeStr(fd, "STACK\n");
            const unsigned long long *sp = reinterpret_cast<const unsigned long long *>(rsp);
            if (!IsBadReadPtr(sp, 64 * sizeof(unsigned long long))) {
                for (int i = 0; i < 64; ++i) {
                    writeHex(fd, sp[i]);
                    writeStr(fd, "\n");
                }
            }
#endif
            CloseHandle(fd);
        }
        TerminateProcess(GetCurrentProcess(), code);
        return EXCEPTION_CONTINUE_SEARCH;
    }

    void abortHandler(int) {
        HANDLE fd = CreateFileA(gReportPath, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (fd != INVALID_HANDLE_VALUE) {
            writeStr(fd, "ABORT\n");
            CloseHandle(fd);
        }
        TerminateProcess(GetCurrentProcess(), 3);
    }

    void installCrashHandler() {
        AddVectoredExceptionHandler(1, crashHandler);
        std::signal(SIGABRT, abortHandler);
    }
#endif

    void mark(const std::string &text) {
        std::string line = text + "\n";
#ifdef _WIN32
        HANDLE fd = CreateFileA(gProgressPath, FILE_APPEND_DATA, FILE_SHARE_READ, nullptr, OPEN_ALWAYS,
                                FILE_ATTRIBUTE_NORMAL, nullptr);
        if (fd == INVALID_HANDLE_VALUE) {
            return;
        }
        writeAll(fd, line.data(), line.size());
        CloseHandle(fd);
#else
        int fd = ::open(gProgressPath, O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (fd < 0) {
            return;
        }
        writeAll(fd, line.data(), line.size());
        ::close(fd);
#endif
    }

    struct HuntConfig {
        bool mcjit = false;
        int loads = 20;
        bool releaseFirst = false;
        bool simulate = false;
        bool inProcess = false;
    };

    // Body of the child process. Always ends the process.
    void runHuntWorkload(const std::string &sbml, const HuntConfig &cfg) {
        std::string tag = envStr("RR_CRASHHUNT_TAG", "hunt_untagged");
        std::string dir = envStr("RR_CRASHHUNT_DIR", std::filesystem::temp_directory_path().string());
        std::snprintf(gReportPath, sizeof(gReportPath), "%s/%s.crash", dir.c_str(), tag.c_str());
        std::snprintf(gProgressPath, sizeof(gProgressPath), "%s/%s.progress", dir.c_str(), tag.c_str());
        if (!cfg.inProcess) {
            installCrashHandler();
            std::string errPath = dir + "/" + tag + ".stderr";
            if (std::freopen(errPath.c_str(), "w", stderr) != nullptr) {
                std::setvbuf(stderr, nullptr, _IONBF, 0);
            }
        }

        // Like the reporter's script: reserve address space so the model's sections end up far apart.
        const int reserveGiB = envInt("RR_CRASHHUNT_RESERVE_GIB", 0);
        if (reserveGiB > 0) {
            const size_t bytes = static_cast<size_t>(reserveGiB) << 30;
#ifdef _WIN32
            VirtualAlloc(nullptr, bytes, MEM_RESERVE, PAGE_NOACCESS);
#else
            mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
#endif
        }

        try {
            Config::setValue(Config::LLVM_BACKEND,
                             cfg.mcjit ? Config::LLVM_BACKEND_VALUES::MCJIT : Config::LLVM_BACKEND_VALUES::LLJIT);
            LoadSBMLOptions opt;
            opt.modelGeneratorOpt = opt.modelGeneratorOpt | LoadSBMLOptions::RECOMPILE;
            if (!cfg.mcjit) {
                opt.setLLJitOptimizationLevel(LoadSBMLOptions::LLJIT_OPTIMIZATION_LEVELS::NONE);
            }

            std::unique_ptr<RoadRunner> previous;
            for (int k = 1; k <= cfg.loads; ++k) {
                mark("load " + std::to_string(k) + " begin");
                if (cfg.releaseFirst) {
                    previous.reset();
                }
                std::unique_ptr<RoadRunner> next = std::make_unique<RoadRunner>(sbml, &opt);
                mark("load " + std::to_string(k) + " constructed");
                if (cfg.simulate) {
                    next->simulate(0, 5, 6);
                    mark("load " + std::to_string(k) + " simulated");
                }
                // the previous runner is destroyed here, after the next one was built
                previous = std::move(next);
            }
            mark("workload done");
            previous.reset();
            mark("teardown done");
        } catch (const std::exception &e) {
            mark(std::string("error: ") + e.what());
            if (cfg.inProcess) {
                std::cout << "[hunt] error: " << e.what() << std::endl;
                return;
            }
            std::_Exit(2);
        }
        if (cfg.inProcess) {
            return;
        }
        std::exit(0);
    }

    // ------------------------------------------------- parent: analysis

    struct Arm {
        std::string name;
        std::map<std::string, std::string> env;
        bool mcjit = false;
        bool aslrOff = false;
    };

    struct ArmResult {
        std::string name;
        int runs = 0;
        int crashes = 0;
        int errors = 0;
        std::vector<std::string> crashNotes;
    };

    std::string slurp(const std::string &path) {
        std::ifstream in(path, std::ios::binary);
        std::stringstream ss;
        ss << in.rdbuf();
        return ss.str();
    }

    // Last few hundred characters of a child's stderr, newlines flattened.
    std::string stderrTail(const std::string &path) {
        std::string text = slurp(path);
        if (text.size() > 400) {
            text = "..." + text.substr(text.size() - 400);
        }
        for (char &c: text) {
            if (c == '\n' || c == '\r') {
                c = ' ';
            }
        }
        return text.empty() ? "(empty)" : text;
    }

    std::string lastLine(const std::string &text) {
        std::istringstream in(text);
        std::string line, last;
        while (std::getline(in, line)) {
            if (!line.empty()) {
                last = line;
            }
        }
        return last;
    }

    // Describes where the faulting PC lies relative to the mappings in a crash report.
    std::string describeCrashReport(const std::string &report) {
        std::istringstream in(report);
        std::string line;
        unsigned long long rip = 0, addr = 0;
        bool haveRip = false, inMaps = false;
        struct Region {
            unsigned long long lo, hi;
            std::string text;
        };
        std::vector<Region> regions;
        std::ostringstream out;
        std::vector<std::string> stack;
        bool inStack = false;
        while (std::getline(in, line)) {
            if (line.rfind("RIP ", 0) == 0) {
                rip = std::stoull(line.substr(4), nullptr, 16);
                haveRip = true;
            } else if (line.rfind("ADDR ", 0) == 0) {
                addr = std::stoull(line.substr(5), nullptr, 16);
            } else if (line == "MAPS") {
                inMaps = true;
                inStack = false;
            } else if (line == "STACK") {
                inMaps = false;
                inStack = true;
            } else if (inMaps) {
                size_t dash = line.find('-');
                size_t space = line.find(' ');
                if (dash != std::string::npos && space != std::string::npos && dash < space) {
                    Region r;
                    r.lo = std::stoull(line.substr(0, dash), nullptr, 16);
                    r.hi = std::stoull(line.substr(dash + 1, space - dash - 1), nullptr, 16);
                    r.text = line;
                    regions.push_back(r);
                }
            } else if (inStack) {
                stack.push_back(line);
            }
        }
        out << "fault address 0x" << std::hex << addr;
        if (haveRip) {
            out << ", pc 0x" << rip;
            const Region *hit = nullptr, *below = nullptr, *above = nullptr;
            for (const Region &r: regions) {
                if (rip >= r.lo && rip < r.hi) {
                    hit = &r;
                } else if (r.hi <= rip && (!below || r.hi > below->hi)) {
                    below = &r;
                } else if (r.lo > rip && (!above || r.lo < above->lo)) {
                    above = &r;
                }
            }
            if (hit) {
                out << " is inside: " << hit->text;
            } else {
                out << " is in no mapping";
                if (below) {
                    out << "; nearest below: " << below->text;
                }
                if (above) {
                    out << "; nearest above: " << above->text;
                }
            }
        }
        out << std::dec << "; " << regions.size() << " mappings, " << stack.size() << " stack words saved";
        return out.str();
    }

    // Whether the child ended through a fatal signal (Linux) or fatal exception / abort (Windows).
    bool diedFromCrash(int status) {
#ifdef _WIN32
        const unsigned code = static_cast<unsigned>(status);
        return code >= 0xC0000000u || code == 3;
#else
        return WIFSIGNALED(status);
#endif
    }

    bool exitedCleanly(int status) {
#ifdef _WIN32
        return status == 0;
#else
        return WIFEXITED(status) && WEXITSTATUS(status) == 0;
#endif
    }

    std::string decodeStatus(int status) {
#ifdef _WIN32
        std::ostringstream out;
        out << "exit code 0x" << std::hex << static_cast<unsigned>(status);
        return out.str();
#else
        if (WIFSIGNALED(status)) {
            return "signal " + std::to_string(WTERMSIG(status));
        }
        if (WIFEXITED(status)) {
            return "exit " + std::to_string(WEXITSTATUS(status));
        }
        return "status " + std::to_string(status);
#endif
    }

    std::vector<Arm> allArms() {
        std::vector<Arm> arms;
        // lljit-baseline is the stock section allocator; lljit-default is the per-object range
        arms.push_back({"lljit-baseline", {{"RR_LLJIT_RESERVE", "0"}}, false, false});
        arms.push_back({"lljit-default", {}, false, false});
        arms.push_back({"lljit-far-rodata", {{"RR_LLJIT_RESERVE", "0"}, {"RR_LLJIT_FAR_RODATA", "1"}}, false, false});
        arms.push_back({"lljit-far-rodata-static-reloc",
                        {{"RR_LLJIT_RESERVE", "0"}, {"RR_LLJIT_FAR_RODATA", "1"}, {"RR_LLJIT_RELOC", "static"}}, false, false});
        arms.push_back({"mcjit", {}, true, false});
        arms.push_back({"lljit-static-reloc", {{"RR_LLJIT_RESERVE", "0"}, {"RR_LLJIT_RELOC", "static"}}, false, false});
        arms.push_back({"lljit-no-compile-thread",
                        {{"RR_LLJIT_RESERVE", "0"}, {"RR_LLJIT_NO_COMPILE_THREAD", "1"}}, false, false});
        arms.push_back({"lljit-no-eh-frames", {{"RR_LLJIT_RESERVE", "0"}, {"RR_LLJIT_NO_EH_FRAMES", "1"}}, false, false});
        arms.push_back({"lljit-aslr-off", {{"RR_LLJIT_RESERVE", "0"}}, false, true});
        arms.push_back({"lljit-reserve",
                        {{"RR_LLJIT_RESERVE", "0"}, {"RR_CRASHHUNT_RESERVE_GIB", "3"}}, false, false});
        arms.push_back({"lljit-default-reserve", {{"RR_CRASHHUNT_RESERVE_GIB", "3"}}, false, false});
        arms.push_back({"lljit-far-rodata-5gib", {{"RR_LLJIT_RESERVE", "0"}, {"RR_LLJIT_FAR_RODATA", "1"}, {"RR_LLJIT_FAR_RODATA_GIB", "5"}},
                        false, false});
        return arms;
    }

    bool armSelected(const std::string &name) {
        std::string filter = envStr("RR_CRASHHUNT_ARMS", "");
        if (filter.empty()) {
            return true;
        }
        std::istringstream in(filter);
        std::string part;
        while (std::getline(in, part, ',')) {
            if (!part.empty() && name.find(part) != std::string::npos) {
                return true;
            }
        }
        return false;
    }

    bool debuggerAttached() {
#ifdef _WIN32
        return IsDebuggerPresent() != FALSE;
#else
        std::ifstream status("/proc/self/status");
        std::string line;
        while (std::getline(status, line)) {
            if (line.rfind("TracerPid:", 0) == 0) {
                return std::atoi(line.c_str() + 10) != 0;
            }
        }
        return false;
#endif
    }

    // Runs one arm once in this process (no death test, no crash handlers) for a debugger.
    void runArmInProcess(const std::string &armName) {
        for (const Arm &arm: allArms()) {
            if (arm.name != armName) {
                continue;
            }
            for (const auto &kv: arm.env) {
                setenv(kv.first.c_str(), kv.second.c_str(), 1);
            }
            setenv("RR_CRASHHUNT_TAG", "hunt_inprocess", 1);
            HuntConfig cfg;
            cfg.mcjit = arm.mcjit;
            cfg.loads = 1;
            cfg.inProcess = true;
            std::cout << "[hunt] running arm " << arm.name << " in this process" << std::endl;
            runHuntWorkload(loadModelText(), cfg);
            std::cout << "[hunt] arm " << arm.name << " finished without a failure" << std::endl;
            for (const auto &kv: arm.env) {
                unsetenv(kv.first.c_str());
            }
        }
    }

#endif // CRASHHUNT_SUPPORTED

} // namespace

// These run without a debugger: they load and simulate through the per-object section range.
TEST(LLJitCrashHunt, InProcessDefault) {
#ifndef CRASHHUNT_SUPPORTED
  GTEST_SKIP() << "The crash hunt needs Linux or Windows.";
#else
  runArmInProcess("lljit-default");
#endif
}

TEST(LLJitCrashHunt, InProcessDefaultReserve) {
#ifndef CRASHHUNT_SUPPORTED
  GTEST_SKIP() << "The crash hunt needs Linux or Windows.";
#else
  runArmInProcess("lljit-default-reserve");
#endif
}

#ifdef CRASHHUNT_SUPPORTED
TEST(LLJitCrashHunt, InProcessFarRodata) {
  if (!debuggerAttached()) {
    GTEST_SKIP() << "Debug this test with a debugger attached to see where the failure happens.";
  }
  runArmInProcess("lljit-far-rodata");
}

TEST(LLJitCrashHunt, InProcessFarRodataStaticReloc) {
  if (!debuggerAttached()) {
    GTEST_SKIP() << "Debug this test with a debugger attached to see where the failure happens.";
  }
  runArmInProcess("lljit-far-rodata-static-reloc");
}
#endif

TEST(LLJitCrashHunt, DISABLED_CompareArms) {
#ifndef CRASHHUNT_SUPPORTED
    GTEST_SKIP() << "The crash hunt needs Linux or Windows.";
#else
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";

    const int runs = envInt("RR_CRASHHUNT_RUNS", 10);
    HuntConfig base;
    base.loads = envInt("RR_CRASHHUNT_LOADS", 3);
    base.releaseFirst = envInt("RR_CRASHHUNT_RELEASE_FIRST", 0) != 0;
    base.simulate = envInt("RR_CRASHHUNT_SIMULATE", 0) != 0;

    const std::string sbml = loadModelText();
    const std::string dir = envStr("RR_CRASHHUNT_DIR", std::filesystem::temp_directory_path().string());
    if (!::testing::internal::InDeathTestChild()) {
        std::cout << "[hunt] crash reports (hunt_<arm>_<run>.crash) and progress files go to " << dir << std::endl;
    }

    std::vector<ArmResult> results;
    int armIndex = 0;
    for (const Arm &arm: allArms()) {
        ++armIndex;
        if (!armSelected(arm.name)) {
            continue;
        }
        ArmResult result;
        result.name = arm.name;

        for (const auto &kv: arm.env) {
            setenv(kv.first.c_str(), kv.second.c_str(), 1);
        }
#ifdef __linux__
        int originalPersona = ::personality(0xffffffff);
#endif
        bool personaChanged = false;
        if (arm.aslrOff) {
#ifdef __linux__
            personaChanged = ::personality(static_cast<unsigned long>(originalPersona) | ADDR_NO_RANDOMIZE) != -1;
#endif
            if (!personaChanged) {
                std::cout << "[hunt] cannot disable ASLR here; skipping arm " << arm.name << std::endl;
                for (const auto &kv: arm.env) {
                    unsetenv(kv.first.c_str());
                }
                continue;
            }
        }

        HuntConfig cfg = base;
        cfg.mcjit = arm.mcjit;

        if (!::testing::internal::InDeathTestChild()) {
            std::cout << "[hunt] arm " << arm.name << " (" << (arm.mcjit ? "MCJIT" : "LLJIT") << ")";
            for (const auto &kv: arm.env) {
                std::cout << " " << kv.first << "=" << kv.second;
            }
            if (arm.aslrOff) {
                std::cout << " ASLR off";
            }
            std::cout << ": " << runs << " runs, " << cfg.loads << " loads each" << std::endl;
        }

        for (int run = 0; run < runs; ++run) {
            std::string tag = "hunt_" + std::to_string(armIndex) + "_" + std::to_string(run);
            setenv("RR_CRASHHUNT_TAG", tag.c_str(), 1);
            std::string progressPath = dir + "/" + tag + ".progress";
            std::string reportPath = dir + "/" + tag + ".crash";
            std::string stderrPath = dir + "/" + tag + ".stderr";
            // the re-executed child replays this loop up to its own run
            if (!::testing::internal::InDeathTestChild()) {
                std::remove(progressPath.c_str());
                std::remove(reportPath.c_str());
                std::remove(stderrPath.c_str());
            }

            int status = -1;
            auto record = [&status](int s) {
                status = s;
                return true;
            };
            EXPECT_EXIT(runHuntWorkload(sbml, cfg), record, "");
            if (::testing::internal::InDeathTestChild()) {
                continue;
            }

            std::string last = lastLine(slurp(progressPath));
            ++result.runs;
            if (exitedCleanly(status)) {
                std::cout << "[hunt] " << arm.name << " run " << run << ": ok" << std::endl;
                std::remove(progressPath.c_str());
                std::remove(stderrPath.c_str());
            } else if (diedFromCrash(status)) {
                ++result.crashes;
                std::string note = "run " + std::to_string(run) + ": " + decodeStatus(status) +
                                   " after '" + last + "'";
                std::string report = slurp(reportPath);
                if (!report.empty()) {
                    note += "; " + describeCrashReport(report) + "; report " + reportPath;
                } else {
                    note += "; no crash report written";
                }
                note += "; stderr: " + stderrTail(stderrPath);
                result.crashNotes.push_back(note);
                std::cout << "[hunt] " << arm.name << " " << note << std::endl;
            } else {
                ++result.errors;
                std::string note = "run " + std::to_string(run) + ": " + decodeStatus(status) +
                                   " after '" + last + "'";
                result.crashNotes.push_back(note);
                std::cout << "[hunt] " << arm.name << " error " << note << std::endl;
            }
        }

#ifdef __linux__
        if (personaChanged) {
            ::personality(static_cast<unsigned long>(originalPersona));
        }
#endif
        for (const auto &kv: arm.env) {
            unsetenv(kv.first.c_str());
        }
        results.push_back(result);
    }

    std::cout << "\n[hunt] runs per arm: " << runs << ", loads per process: " << base.loads
              << ", release first: " << base.releaseFirst << ", simulate: " << base.simulate << "\n";
    std::cout << std::left << std::setw(26) << "arm" << std::right << std::setw(6) << "runs"
              << std::setw(9) << "crashes" << std::setw(8) << "errors" << std::setw(8) << "rate"
              << std::setw(9) << "95% UB" << std::setw(16) << "p vs baseline" << "\n";
    const ArmResult *baseline = nullptr;
    for (const ArmResult &r: results) {
        if (r.name == "lljit-baseline") {
            baseline = &r;
        }
    }
    bool anyCrash = false;
    for (const ArmResult &r: results) {
        anyCrash = anyCrash || r.crashes > 0;
        double rate = r.runs ? static_cast<double>(r.crashes) / r.runs : 0.0;
        std::cout << std::left << std::setw(26) << r.name << std::right << std::setw(6) << r.runs
                  << std::setw(9) << r.crashes << std::setw(8) << r.errors << std::fixed
                  << std::setprecision(3) << std::setw(8) << rate << std::setw(9)
                  << upperBound95(r.crashes, r.runs);
        if (baseline && &r != baseline) {
            std::cout << std::setw(16)
                      << fisherExact(r.crashes, r.runs - r.crashes, baseline->crashes,
                                     baseline->runs - baseline->crashes);
        } else {
            std::cout << std::setw(16) << "-";
        }
        std::cout << "\n";
    }
    std::cout << std::flush;

    for (const ArmResult &r: results) {
        EXPECT_EQ(r.runs, runs) << "arm " << r.name;
    }
    if (envInt("RR_CRASHHUNT_FAIL_ON_CRASH", 0) != 0) {
        EXPECT_FALSE(anyCrash) << "at least one run crashed";
    }
#endif
}

#ifdef CRASHHUNT_SUPPORTED
#endif

#ifdef CRASHHUNT_SUPPORTED
TEST(LLJitCrashHunt, DISABLED_InProcessFarRodata5GiB) {
    if (!debuggerAttached()) {
        GTEST_SKIP() << "Debug this test with a debugger attached to see where the failure happens.";
    }
    runArmInProcess("lljit-far-rodata-5gib");
}
#endif
