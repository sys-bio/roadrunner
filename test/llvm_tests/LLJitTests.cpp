//
// Created by Ciaran on 25/10/2021.
//

#include <GetValueCodeGenBase.h>
#include "llvm/LLJit.h"
#include "gtest/gtest.h"
#include "TestModelFactory.h"
#include "rrLogger.h"
#include "rrRoadRunnerOptions.h"
#include "JitTests.h"
#include "rrConfig.h"
#include "rrUtils.h"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include "llvm/SBMLModelObjectCache.h"

using namespace rr;
using namespace rrllvm;

class LLJitTests : public JitTests {
public:
    LLJitTests() {
        Logger::setLevel(Logger::LOG_ERROR);
    };

};

using fibonacciFnPtr = int (*)(int);

TEST_F(LLJitTests, ModuleNonOwningNotNull) {
    LoadSBMLOptions opt;
    LLJit llJit(opt.modelGeneratorOpt);
    ASSERT_TRUE(llJit.getModuleNonOwning());
}

TEST_F(LLJitTests, ContextNonOwningNotNull) {
    LoadSBMLOptions opt;
    LLJit llJit(opt.modelGeneratorOpt);
    ASSERT_TRUE(llJit.getContextNonOwning());
}

TEST_F(LLJitTests, BuilderNonOwningNotNull) {
    LoadSBMLOptions opt;
    LLJit llJit(opt.modelGeneratorOpt);
    ASSERT_TRUE(llJit.getBuilderNonOwning());
}

TEST_F(LLJitTests, CreateJittedFibonacci) {
    // maybe the module and shouldnt be owned by the jit?:?
    LoadSBMLOptions opt;
    LLJit llJit(opt.modelGeneratorOpt);
    CreateFibFunction(llJit.getModuleNonOwning());
    std::cout << llJit.emitToString();
    llJit.addIRModule();
    fibonacciFnPtr fibPtr = (int (*)(int)) llJit.lookupFunctionAddress("fib");
    ASSERT_EQ(fibPtr(4), 3);
}

TEST_F(LLJitTests, FibonacciCached) {
    // maybe the module and shouldnt be owned by the jit?:?
    LoadSBMLOptions opt;
    LLJit llJit(opt.modelGeneratorOpt);
    llJit.setModuleIdentifier("fibMod");
    CreateFibFunction(llJit.getModuleNonOwning());
//    std::cout << llJit.emitToString();
    llJit.addIRModule();
    fibonacciFnPtr fibPtr1 = (int (*)(int)) llJit.lookupFunctionAddress("fib");
    ASSERT_EQ(fibPtr1(4), 3);

    std::unique_ptr<llvm::LLVMContext> ctx = std::make_unique<llvm::LLVMContext>();
    auto mod = std::make_unique<llvm::Module>("fibMod", *ctx);
    mod->setModuleIdentifier("fibMod");

    std::unique_ptr<llvm::MemoryBuffer> obj = SBMLModelObjectCache::getObjectCache().getObject(mod.get());
    LLJit llJit2(opt.modelGeneratorOpt);
    llJit2.addObjectFile(std::move(obj));

    fibonacciFnPtr fibPtr2 = (int (*)(int)) llJit2.lookupFunctionAddress("fib");
    ASSERT_EQ(fibPtr2(4), 3);

}

TEST_F(LLJitTests, LoadPowerFunction) {
    LoadSBMLOptions opt;
    LLJit llJit(opt.modelGeneratorOpt);
    auto fnPtr = (powFnTy) llJit.lookupFunctionAddress("pow");
    ASSERT_EQ(fnPtr(2, 2), 4);
}

TEST_F(LLJitTests, LoadQuotientFunction) {
    LoadSBMLOptions opt;
    LLJit llJit(opt.modelGeneratorOpt);
    auto fnPtr = (quotientFnTy) llJit.lookupFunctionAddress("quotient");
    ASSERT_EQ(fnPtr(4, 2), 2);
}

TEST_F(LLJitTests, Loadrr_minFunction) {
    LoadSBMLOptions opt;
    LLJit llJit(opt.modelGeneratorOpt);
    auto fnPtr = (quotientFnTy) llJit.lookupFunctionAddress("rr_min");
    ASSERT_EQ(fnPtr(4, 2), 2);
}

TEST_F(LLJitTests, Loadrr_csr_matrix_get_nzFunction) {
    LoadSBMLOptions opt;
    LLJit llJit(opt.modelGeneratorOpt);
    auto addr = llJit.lookupFunctionAddress("csr_matrix_get_nz");
    ASSERT_TRUE(addr);
}

TEST_F(LLJitTests, SetModuleIdentifier) {
    LoadSBMLOptions opt;
    LLJit llJit(opt.modelGeneratorOpt);
    llJit.setModuleIdentifier("FibMod");
    ASSERT_STREQ("FibMod", llJit.getModuleNonOwning()->getModuleIdentifier().c_str());
}


/**
 * Not concerned with simulation accuracy here, only that the sbml
 * compiles and the model simulates.
 */
TEST_F(LLJitTests, CheckModelSimulates) {
//    rr::Config::setValue(rr::Config::LLVM_BACKEND, rr::Config::LLVM_BACKEND_VALUES::MCJIT);
    rr::Config::setValue(rr::Config::LLVM_BACKEND, rr::Config::LLVM_BACKEND_VALUES::LLJIT);
    RoadRunner rr(OpenLinearFlux().str());
    auto data = rr.simulate(0, 10, 11);
    ASSERT_EQ(typeid(*data), typeid(ls::Matrix<double>));
}


TEST_F(LLJitTests, GetObjectFromCacheUsingOriginalString) {
//    rr::Config::setValue(rr::Config::LLVM_BACKEND, rr::Config::LLVM_BACKEND_VALUES::MCJIT);
    rr::Config::setValue(rr::Config::LLVM_BACKEND, rr::Config::LLVM_BACKEND_VALUES::LLJIT);
    RoadRunner rr(OpenLinearFlux().str());
    auto &cache = rrllvm::SBMLModelObjectCache::getObjectCache();
    cache.inspect();
    const std::string &sbml = rr.getSBML();
    std::string md5 = rr::getSBMLMD5(OpenLinearFlux().str(), LoadSBMLOptions().modelGeneratorOpt);
    LLVMContext ctx;
    auto m = std::make_unique<llvm::Module>(md5, ctx);

    ASSERT_TRUE(m && "module loaded from LLJit cache is nullptr");
    /**
     * Test will segfault here if no model was found.
     */
    std::unique_ptr<llvm::MemoryBuffer> objectBuf = cache.getObject(m.get());
    ASSERT_TRUE(objectBuf);

    Expected<std::unique_ptr<llvm::object::ObjectFile>>
            objFile = llvm::object::ObjectFile::createObjectFile(objectBuf->getMemBufferRef());

    if (!objFile) {
        //LS DEBUG:  find a way to get the text out of the error.
        auto err = objFile.takeError();
        std::string s = "LLVM object supposed to be file, but is not.";
        rrLog(Logger::LOG_FATAL) << s;
        throw_llvm_exception(s);
    }
    LLJit llJit(LoadSBMLOptions().modelGeneratorOpt);
    llJit.addObjectFile(std::move(*objFile));



    // todo more assertions?

}

/**
 * This fails at the moment because we've modified the sbml md5
 * post loading, so the original md5 is the key to the cache while
 * rr.getSBML() now contains a different hash.
 */
TEST_F(LLJitTests, DISABLED_GetObjectFromCacheUsingRoadRunnerMethods) {
//    rr::Config::setValue(rr::Config::LLVM_BACKEND, rr::Config::LLVM_BACKEND_VALUES::MCJIT);
    rr::Config::setValue(rr::Config::LLVM_BACKEND, rr::Config::LLVM_BACKEND_VALUES::LLJIT);
    RoadRunner rr(OpenLinearFlux().str());
    auto &cache = rrllvm::SBMLModelObjectCache::getObjectCache();
    cache.inspect();
    const std::string &sbml = rr.getSBML();
    std::string md5 = rr::getMD5(rr.getSBML());
    LLVMContext ctx;
    auto m = std::make_unique<llvm::Module>(md5, ctx);

    /**
     * Test will segfault here if no model was found.
     */
    std::unique_ptr<llvm::MemoryBuffer> objectBuf = cache.getObject(m.get());
    ASSERT_TRUE(objectBuf);

    Expected<std::unique_ptr<llvm::object::ObjectFile>>
            objFile = llvm::object::ObjectFile::createObjectFile(objectBuf->getMemBufferRef());

    if (!objFile) {
        //LS DEBUG:  find a way to get the text out of the error.
        auto err = objFile.takeError();
        std::string s = "LLVM object supposed to be file, but is not.";
        rrLog(Logger::LOG_FATAL) << s;
        throw_llvm_exception(s);
    }
    LLJit llJit(LoadSBMLOptions().modelGeneratorOpt);
    llJit.addObjectFile(std::move(*objFile));
}

namespace {
    // Minimal model with a unique id so each call yields a model that is not yet in any cache.
    std::string makeUniqueModel() {
        static int counter = 0;
        std::ostringstream s;
        s << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
             "<sbml xmlns=\"http://www.sbml.org/sbml/level3/version1/core\" level=\"3\" version=\"1\">\n"
             "<model id=\"cacheKeyTest_"
          << std::chrono::steady_clock::now().time_since_epoch().count() << "_" << ++counter << "\">\n"
             "<listOfCompartments><compartment id=\"c\" spatialDimensions=\"3\" size=\"1\" constant=\"true\"/></listOfCompartments>\n"
             "<listOfSpecies><species id=\"S1\" compartment=\"c\" initialConcentration=\"1\" "
             "hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/></listOfSpecies>\n"
             "<listOfParameters><parameter id=\"k\" value=\"0.1\" constant=\"true\"/></listOfParameters>\n"
             "<listOfReactions><reaction id=\"R1\" reversible=\"false\" fast=\"false\">\n"
             "<listOfReactants><speciesReference species=\"S1\" stoichiometry=\"1\" constant=\"true\"/></listOfReactants>\n"
             "<kineticLaw><math xmlns=\"http://www.w3.org/1998/Math/MathML\"><apply><times/><ci>k</ci><ci>S1</ci></apply></math></kineticLaw>\n"
             "</reaction></listOfReactions>\n</model>\n</sbml>\n";
        return s.str();
    }

    size_t objectCacheSize() {
        return rrllvm::SBMLModelObjectCache::getObjectCache().inspect().size();
    }
}

/**
 * Control: loading the same model twice with the same optimization level
 * should add exactly one object to the cache.
 */
TEST_F(LLJitTests, ObjectCacheSameModelSameOptLevelAddsOneEntry) {
    rr::Config::setValue(rr::Config::LLVM_BACKEND, rr::Config::LLVM_BACKEND_VALUES::LLJIT);
    std::string sbml = makeUniqueModel();
    LoadSBMLOptions opt;
    opt.setLLJitOptimizationLevel(LoadSBMLOptions::NONE);
    size_t before = objectCacheSize();
    RoadRunner rr1(sbml, &opt);
    RoadRunner rr2(sbml, &opt);
    ASSERT_EQ(objectCacheSize() - before, 1u);
}

/**
 * The same model loaded at different optimization levels must not share
 * a cached object, since the levels produce different machine code.
 */
TEST_F(LLJitTests, ObjectCacheSameModelDifferentOptLevelsAddsSeparateEntries) {
    rr::Config::setValue(rr::Config::LLVM_BACKEND, rr::Config::LLVM_BACKEND_VALUES::LLJIT);
    std::string sbml = makeUniqueModel();
    LoadSBMLOptions optNone;
    optNone.setLLJitOptimizationLevel(LoadSBMLOptions::NONE);
    LoadSBMLOptions optAggressive;
    optAggressive.setLLJitOptimizationLevel(LoadSBMLOptions::AGGRESSIVE);
    size_t before = objectCacheSize();
    RoadRunner rr1(sbml, &optNone);
    RoadRunner rr2(sbml, &optAggressive);
    ASSERT_EQ(objectCacheSize() - before, 2u);
}

/**
 * Same as above, but the first model is destroyed before the second is
 * loaded, so only the object cache (not the live model resources) could
 * cause reuse.
 */
TEST_F(LLJitTests, ObjectCacheDifferentOptLevelsAfterFirstModelDestroyed) {
    rr::Config::setValue(rr::Config::LLVM_BACKEND, rr::Config::LLVM_BACKEND_VALUES::LLJIT);
    std::string sbml = makeUniqueModel();
    LoadSBMLOptions optNone;
    optNone.setLLJitOptimizationLevel(LoadSBMLOptions::NONE);
    LoadSBMLOptions optAggressive;
    optAggressive.setLLJitOptimizationLevel(LoadSBMLOptions::AGGRESSIVE);
    size_t before = objectCacheSize();
    {
        RoadRunner rr1(sbml, &optNone);
    }
    {
        RoadRunner rr2(sbml, &optAggressive);
    }
    ASSERT_EQ(objectCacheSize() - before, 2u);
}

/**
 * Zero compile threads means compiling on the calling thread; it must still load and simulate.
 */
TEST_F(LLJitTests, ZeroCompileThreadsLoadsAndSimulates) {
    rr::Config::setValue(rr::Config::LLVM_BACKEND, rr::Config::LLVM_BACKEND_VALUES::LLJIT);
    rr::Config::setValue(rr::Config::LLJIT_NUM_THREADS, 0);
    const ls::DoubleMatrix *data = nullptr;
    {
        RoadRunner rr(makeUniqueModel());
        data = rr.simulate(0, 10, 11);
    }
    rr::Config::setValue(rr::Config::LLJIT_NUM_THREADS, 1);
    ASSERT_TRUE(data != nullptr);
}

namespace {
    int envIntOr(const char *name, int fallback) {
        const char *v = std::getenv(name);
        return v ? std::atoi(v) : fallback;
    }

    // One species with a single reaction whose rate law is a left-nested sum of the given
    // depth: (((k*S1 + k*S1) + k*S1) + ...) .
    std::string makeNestedRateLawModel(int depth) {
        std::ostringstream s;
        s << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
             "<sbml xmlns=\"http://www.sbml.org/sbml/level3/version1/core\" level=\"3\" version=\"1\">\n"
             "<model id=\"nested_" << depth << "_" << std::chrono::steady_clock::now().time_since_epoch().count() << "\">\n"
             "<listOfCompartments><compartment id=\"c\" spatialDimensions=\"3\" size=\"1\" constant=\"true\"/></listOfCompartments>\n"
             "<listOfSpecies><species id=\"S1\" compartment=\"c\" initialConcentration=\"1\" "
             "hasOnlySubstanceUnits=\"false\" boundaryCondition=\"false\" constant=\"false\"/></listOfSpecies>\n"
             "<listOfParameters><parameter id=\"k\" value=\"0.001\" constant=\"true\"/></listOfParameters>\n"
             "<listOfReactions><reaction id=\"R1\" reversible=\"false\" fast=\"false\">\n"
             "<listOfReactants><speciesReference species=\"S1\" stoichiometry=\"1\" constant=\"true\"/></listOfReactants>\n"
             "<kineticLaw><math xmlns=\"http://www.w3.org/1998/Math/MathML\">";
        const std::string term = "<apply><times/><ci>k</ci><ci>S1</ci></apply>";
        for (int i = 0; i < depth; ++i) {
            s << "<apply><plus/>";
        }
        s << term;
        for (int i = 0; i < depth; ++i) {
            s << term << "</apply>";
        }
        s << "</math></kineticLaw>\n</reaction></listOfReactions>\n</model>\n</sbml>\n";
        return s.str();
    }
}

/**
 * Checks whether compiling a deeply nested rate law works when LLVM compiles on the calling
 * thread (LLJIT_NUM_THREADS = 0), compared with a thread pool. A stack overflow kills the
 * process, so each depth is printed before it is tried; run this once per setting:
 *   RR_NESTING_THREADS  LLJIT_NUM_THREADS to use (default 0)
 *   RR_NESTING_OPT      LLJIT_OPTIMIZATION_LEVEL, 0=NONE 1=LESS 2=DEFAULT 3=AGGRESSIVE (default 0)
 *   RR_NESTING_DEPTHS   comma separated nesting depths (default 100,500,1000,2000,5000,10000,20000)
 * Run with --gtest_also_run_disabled_tests --gtest_filter=*DeeplyNested*
 * If 0 and 1 threads both fail at the same depth, the limit comes from SBML parsing or IR
 * generation, which run on the calling thread either way; if 0 fails at a lower depth than 1,
 * the LLVM backend needs more stack than the calling thread has.
 */
TEST_F(LLJitTests, DISABLED_DeeplyNestedRateLawCompiles) {
    // The fixture turns on debug logging, which would bury the depth lines.
    Logger::setLevel(Logger::LOG_ERROR);
    const int threads = envIntOr("RR_NESTING_THREADS", 1);
    const int opt = envIntOr("RR_NESTING_OPT", 0);
    const char *depthsEnv = std::getenv("RR_NESTING_DEPTHS");
    std::istringstream depths(depthsEnv ? depthsEnv : "560,561,562,563,564,565,570,575,580,600,700,800,900,950");

    std::cout << "LLJIT_NUM_THREADS = " << threads << ", LLJIT_OPTIMIZATION_LEVEL = " << opt << std::endl;
    std::string item;
    while (std::getline(depths, item, ',')) {
        if (item.empty()) {
            continue;
        }
        const int depth = std::atoi(item.c_str());
        rr::Config::resetToDefaults();
        rr::Config::setValue(rr::Config::LLVM_BACKEND, rr::Config::LLVM_BACKEND_VALUES::LLJIT);
        rr::Config::setValue(rr::Config::LLJIT_NUM_THREADS, threads);
        rr::Config::setValue(rr::Config::LLJIT_OPTIMIZATION_LEVEL, opt);
        std::cout << "depth " << depth << ": " << std::flush;
        try {
            RoadRunner rr(makeNestedRateLawModel(depth));
            rr.simulate(0, 1, 2);
            std::cout << "ok" << std::endl;
        } catch (const std::exception &e) {
            std::cout << "exception: " << e.what() << std::endl;
        }
    }
    rr::Config::resetToDefaults();
}

namespace {
    // Loads one fresh model with `first`, then with `second`, keeping both alive, and returns how
    // many objects that added to the object cache. The second model is simulated, which fails
    // with a missing symbol if it was given the first model's incompatible compiled code.
    size_t objectsAddedByLoadingWith(LoadSBMLOptions &first, LoadSBMLOptions &second) {
        const std::string sbml = makeUniqueModel();
        const size_t before = objectCacheSize();
        RoadRunner rr1(sbml, &first);
        RoadRunner rr2(sbml, &second);
        rr2.simulate(0, 1, 2);
        return objectCacheSize() - before;
    }
}

TEST_F(LLJitTests, ObjectCacheMutableInitialConditionsAddsSeparateEntry) {
    rr::Config::setValue(rr::Config::LLVM_BACKEND, rr::Config::LLVM_BACKEND_VALUES::LLJIT);
    LoadSBMLOptions immutable;
    immutable.modelGeneratorOpt &= ~LoadSBMLOptions::MUTABLE_INITIAL_CONDITIONS;
    LoadSBMLOptions mutableICs;
    mutableICs.modelGeneratorOpt |= LoadSBMLOptions::MUTABLE_INITIAL_CONDITIONS;
    ASSERT_EQ(objectsAddedByLoadingWith(immutable, mutableICs), 2u);
}

TEST_F(LLJitTests, ObjectCacheReadOnlyAddsSeparateEntry) {
    rr::Config::setValue(rr::Config::LLVM_BACKEND, rr::Config::LLVM_BACKEND_VALUES::LLJIT);
    LoadSBMLOptions writable;
    LoadSBMLOptions readOnly;
    readOnly.modelGeneratorOpt |= LoadSBMLOptions::READ_ONLY;
    ASSERT_EQ(objectsAddedByLoadingWith(readOnly, writable), 2u);
}

TEST_F(LLJitTests, ReadOnlyIgnoresMutableInitialConditions) {
    rr::Config::setValue(rr::Config::LLVM_BACKEND, rr::Config::LLVM_BACKEND_VALUES::LLJIT);
    LoadSBMLOptions opt;
    opt.modelGeneratorOpt |= LoadSBMLOptions::READ_ONLY | LoadSBMLOptions::MUTABLE_INITIAL_CONDITIONS;
    RoadRunner rr(makeUniqueModel(), &opt);
    ASSERT_NO_THROW(rr.simulate(0, 1, 2));
}

TEST_F(LLJitTests, ObjectCachePermissiveSettingAddsSeparateEntry) {
    rr::Config::setValue(rr::Config::LLVM_BACKEND, rr::Config::LLVM_BACKEND_VALUES::LLJIT);
    const std::string sbml = makeUniqueModel();
    const size_t before = objectCacheSize();
    rr::Config::setValue(rr::Config::LOADSBMLOPTIONS_PERMISSIVE, true);
    RoadRunner rr1(sbml);
    rr::Config::setValue(rr::Config::LOADSBMLOPTIONS_PERMISSIVE, false);
    RoadRunner rr2(sbml);
    rr::Config::setValue(rr::Config::LOADSBMLOPTIONS_PERMISSIVE, true);
    ASSERT_EQ(objectCacheSize() - before, 2u);
}

/**
 * Each load option that changes the generated code must change the key; RECOMPILE only changes
 * whether the cache is consulted, so it must not.
 */
TEST_F(LLJitTests, SBMLMD5KeyDependsOnCodeGenerationOptions) {
    rr::Config::setValue(rr::Config::LOADSBMLOPTIONS_PERMISSIVE, true);
    const std::string sbml = makeUniqueModel();
    const int base = (int) LoadSBMLOptions().modelGeneratorOpt & ~(int) LoadSBMLOptions::MUTABLE_INITIAL_CONDITIONS;
    const std::string baseKey = rr::getSBMLMD5(sbml, base);

    EXPECT_NE(baseKey, rr::getSBMLMD5(sbml, base | LoadSBMLOptions::CONSERVED_MOIETIES));
    EXPECT_NE(baseKey, rr::getSBMLMD5(sbml, base | LoadSBMLOptions::MUTABLE_INITIAL_CONDITIONS));
    EXPECT_NE(baseKey, rr::getSBMLMD5(sbml, base | LoadSBMLOptions::READ_ONLY));
    EXPECT_EQ(baseKey, rr::getSBMLMD5(sbml, base | LoadSBMLOptions::RECOMPILE));

    LoadSBMLOptions none;
    none.setLLJitOptimizationLevel(LoadSBMLOptions::NONE);
    LoadSBMLOptions aggressive;
    aggressive.setLLJitOptimizationLevel(LoadSBMLOptions::AGGRESSIVE);
    EXPECT_NE(rr::getSBMLMD5(sbml, (int) none.modelGeneratorOpt),
              rr::getSBMLMD5(sbml, (int) aggressive.modelGeneratorOpt));

    rr::Config::setValue(rr::Config::LOADSBMLOPTIONS_PERMISSIVE, false);
    EXPECT_NE(baseKey, rr::getSBMLMD5(sbml, base));
    rr::Config::setValue(rr::Config::LOADSBMLOPTIONS_PERMISSIVE, true);
}


/**
 * Sets the LLJit backend and restores the configuration values that the
 * per-LLJit object cache tests below change.
 */
class LLJitObjectCacheTests : public LLJitTests {
public:
    LLJitObjectCacheTests() {
        saved = {Config::LLVM_BACKEND, Config::LOADSBMLOPTIONS_RECOMPILE};
        for (Config::Keys key: saved) {
            savedValues.push_back(Config::getValue(key));
        }
        Config::setValue(Config::LLVM_BACKEND, Config::LLJIT);
        Config::setValue(Config::LOADSBMLOPTIONS_RECOMPILE, false);
    }

    ~LLJitObjectCacheTests() override {
        for (size_t i = 0; i < saved.size(); ++i) {
            Config::setValue(saved[i], savedValues[i]);
        }
    }

    /**
     * @brief the object in the object cache for the model sbml when loaded with the
     * current configuration.
     */
    static SBMLModelObjectCache::ObjectPtr cachedObject(const std::string &sbml) {
        return SBMLModelObjectCache::getObjectCache().find(
                rr::getSBMLMD5(sbml, LoadSBMLOptions().modelGeneratorOpt));
    }

    std::vector<Config::Keys> saved;
    std::vector<Setting> savedValues;
};

/**
 * Without RECOMPILE, a model loaded again after the first one is gone uses
 * the object that the first one compiled.
 */
TEST_F(LLJitObjectCacheTests, ReusesObjectWithoutRecompile) {
    std::string sbml = OpenLinearFlux().str();
    {
        RoadRunner rr(sbml);
    }
    SBMLModelObjectCache::ObjectPtr first = cachedObject(sbml);
    ASSERT_TRUE(first);
    RoadRunner rr(sbml);
    ASSERT_EQ(first.get(), cachedObject(sbml).get());
    ASSERT_NO_THROW(rr.simulate(0, 10, 11));
}

/**
 * With RECOMPILE, the model is compiled again, and its object replaces the one
 * in the object cache (issue 1420).
 */
TEST_F(LLJitObjectCacheTests, RecompileDoesNotReuseCachedObject) {
    std::string sbml = OpenLinearFlux().str();
    Config::setValue(Config::LOADSBMLOPTIONS_RECOMPILE, true);
    RoadRunner rr1(sbml);
    SBMLModelObjectCache::ObjectPtr first = cachedObject(sbml);
    ASSERT_TRUE(first);
    RoadRunner rr2(sbml);
    SBMLModelObjectCache::ObjectPtr second = cachedObject(sbml);
    ASSERT_TRUE(second);
    ASSERT_NE(first.get(), second.get());
    ASSERT_NO_THROW(rr1.simulate(0, 10, 11));
    ASSERT_NO_THROW(rr2.simulate(0, 10, 11));
}

/**
 * The state of a model holds the object of that model, also after another
 * model replaced the object in the object cache.
 */
TEST_F(LLJitObjectCacheTests, SaveStateUsesTheObjectOfTheModel) {
    std::string sbml = OpenLinearFlux().str();
    Config::setValue(Config::LOADSBMLOPTIONS_RECOMPILE, true);
    RoadRunner rr1(sbml);
    RoadRunner rr2(sbml);
    RoadRunner rr3;
    // loadStateS deletes the stream
    rr3.loadStateS(rr1.saveStateS());
    auto expected = rr1.simulate(0, 10, 11);
    auto actual = rr3.simulate(0, 10, 11);
    ASSERT_EQ(expected->numRows(), actual->numRows());
    for (int i = 0; i < actual->numRows(); ++i) {
        for (int j = 0; j < actual->numCols(); ++j) {
            ASSERT_NEAR((*expected)(i, j), (*actual)(i, j), 1e-10);
        }
    }
}
