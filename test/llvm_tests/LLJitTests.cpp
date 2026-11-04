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

using namespace rr;
using namespace rrllvm;

class LLJitTests : public JitTests {
public:
    LLJitTests() {
        Logger::setLevel(Logger::LOG_DEBUG);
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


/**
 * Sets the LLJit backend and restores the configuration values that the object
 * cache tests below change.
 */
class LLJitObjectCacheTests : public LLJitTests {
public:
    LLJitObjectCacheTests() {
        saved = {Config::LLVM_BACKEND, Config::LLJIT_OPTIMIZATION_LEVEL,
                 Config::LOADSBMLOPTIONS_RECOMPILE,
                 Config::LOADSBMLOPTIONS_MUTABLE_INITIAL_CONDITIONS};
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
 * The optimization level is part of the key of an object, so a model loaded with
 * another level is compiled with that level (issue 1420).
 */
TEST_F(LLJitObjectCacheTests, OptimizationLevelIsPartOfTheKey) {
    std::string sbml = OpenLinearFlux().str();
    Config::setValue(Config::LLJIT_OPTIMIZATION_LEVEL, Config::NONE);
    std::string keyNone = rr::getSBMLMD5(sbml, LoadSBMLOptions().modelGeneratorOpt);
    {
        RoadRunner rr(sbml);
    }
    Config::setValue(Config::LLJIT_OPTIMIZATION_LEVEL, Config::AGGRESSIVE);
    std::string keyAggressive = rr::getSBMLMD5(sbml, LoadSBMLOptions().modelGeneratorOpt);
    ASSERT_NE(keyNone, keyAggressive);
    {
        RoadRunner rr(sbml);
    }
    ASSERT_TRUE(SBMLModelObjectCache::getObjectCache().find(keyNone));
    ASSERT_TRUE(SBMLModelObjectCache::getObjectCache().find(keyAggressive));
}

/**
 * The model cache and the object cache are keyed by the load options too, so
 * a model with other load options gets its own code (issue 1420). Before, the
 * second model used the object of the first one, and loading it crashed because
 * a function of the second model was not in that object.
 */
TEST_F(LLJitObjectCacheTests, LoadOptionsArePartOfTheKey) {
    std::string sbml = OpenLinearFlux().str();
    Config::setValue(Config::LOADSBMLOPTIONS_MUTABLE_INITIAL_CONDITIONS, true);
    RoadRunner mutableRR(sbml);
    Config::setValue(Config::LOADSBMLOPTIONS_MUTABLE_INITIAL_CONDITIONS, false);
    RoadRunner immutableRR(sbml);
    {
        // with the first model gone, the object cache would be the only one to share code
        RoadRunner rr(sbml);
        ASSERT_NO_THROW(rr.simulate(0, 10, 11));
    }
    ASSERT_NO_THROW(mutableRR.simulate(0, 10, 11));
    ASSERT_NO_THROW(immutableRR.simulate(0, 10, 11));
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
