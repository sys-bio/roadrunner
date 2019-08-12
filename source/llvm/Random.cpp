/*
 * Random.cpp
 *
 *  Created on: Sep 23, 2014
 *      Author: andy
 */

#include "Random.h"
#include "ModelGeneratorContext.h"
#include "LLVMIncludes.h"
#include "rrLogger.h"
#include "rrConfig.h"
#include "rrUtils.h"
#include <stdint.h>


#if INTPTR_MAX == INT32_MAX
    #define RR_32BIT
#elif INTPTR_MAX == INT64_MAX
   #define RR_64BIT
#else
    #error "Environment not 32 or 64-bit."
#endif

using rr::Logger;
using rr::Config;
using rr::getMicroSeconds;

using namespace llvm;

namespace rrllvm
{

typedef cxx11_ns::normal_distribution<double> NormalDist;

typedef cxx11_ns::poisson_distribution<int> PoissonDist;

typedef cxx11_ns::exponential_distribution<double> ExponentialDist;

typedef cxx11_ns::lognormal_distribution<double> LognormalDist;

static int randomCount = 0;

/**
 * random uniform distribution
 */
static double distrib_uniform(Random *random, double _min, double _max);

static double distrib_normal(Random* random, double mu, double sigma);

static double distrib_poisson(Random* random, double lambda);

static double distrib_exponential(Random* random, double lambda);

static double distrib_lognormal(Random* random, double mu, double sigma);

static Function* createGlobalMappingFunction(const char* funcName,
        llvm::FunctionType *funcType, Module *module);

static void addGlobalMappings(const ModelGeneratorContext& ctx);

static int64_t defaultSeed()
{
    int64_t seed = Config::getValue(Config::RANDOM_SEED).convert<int>();
    if (seed < 0)
    {
        // system time in mirsoseconds since 1970
        seed = getMicroSeconds();
    }
    return seed;
}

Random::Random(ModelGeneratorContext& ctx)
{
    addGlobalMappings(ctx);
    setRandomSeed(defaultSeed());
    randomCount++;
}

Random::Random(const Random& other)
{
    *this = other;
    setRandomSeed(defaultSeed());
    randomCount++;
}

Random::Random()
{
    setRandomSeed(defaultSeed());
    randomCount++;
}

Random& Random::operator =(const Random& rhs)
{
    engine = rhs.engine;
    return *this;
}

Function* createGlobalMappingFunction(const char* funcName,
        llvm::FunctionType *funcType, Module *module)
{
    return Function::Create(funcType, Function::InternalLinkage, funcName, module);
}

void addGlobalMappings(const ModelGeneratorContext& ctx)
{
    llvm::Module *module = ctx.getModule();
    LLVMContext& context = module->getContext();
    llvm::ExecutionEngine *executionEngine = &ctx.getExecutionEngine();
    Type *double_type = Type::getDoubleTy(context);
    Type *int_type = Type::getInt32Ty(context);


    // LLVM does not appear to have a true void ptr, so just use a pointer
    // to a byte, pointers are all the same size anyway.
    // used for the LLVMModelData::random which is not accessed by
    // generated llvm code anyway.
    // see also, llvm::StructType *ModelDataIRBuilder::createModelDataStructType(...)
    Type *voidPtrType = Type::getInt8PtrTy(context);

    Type* args_void_double_double[] = { voidPtrType, double_type, double_type };

    Type* args_void_double[] = { voidPtrType, double_type };



    executionEngine->addGlobalMapping(
            createGlobalMappingFunction("rr_distrib_uniform",
                    FunctionType::get(double_type, args_void_double_double, false), module),
                        (void*) distrib_uniform);

    executionEngine->addGlobalMapping(
            createGlobalMappingFunction("rr_distrib_normal",
                    FunctionType::get(double_type, args_void_double_double, false), module),
                        (void*) distrib_normal);
    
    executionEngine->addGlobalMapping(
        createGlobalMappingFunction("rr_distrib_poisson",
            FunctionType::get(double_type, args_void_double, false), module),
            (void*)distrib_poisson);

    executionEngine->addGlobalMapping(
        createGlobalMappingFunction("rr_distrib_exponential",
            FunctionType::get(double_type, args_void_double, false), module),
            (void*)distrib_exponential);

    executionEngine->addGlobalMapping(
        createGlobalMappingFunction("rr_distrib_lognormal",
            FunctionType::get(double_type, args_void_double_double, false), module),
            (void*)distrib_lognormal);

}

double distrib_uniform(Random *random, double _min, double _max)
{
    Log(Logger::LOG_DEBUG) << "distrib_uniform("
            << static_cast<void*>(random)
            << ", " << _min << ", " << _max << ")";

#ifdef CXX11_RANDOM
    cxx11_ns::uniform_real_distribution<double> dist(_min, _max);
#else
    cxx11_ns::uniform_real<double> dist(_min, _max);
#endif
    return dist(*random);
}

double distrib_normal(Random* random, double mu, double sigma)
{
    Log(Logger::LOG_DEBUG) << "distrib_normal("
            << static_cast<void*>(random)
            << ", " << mu << ", " << sigma << ")";

    NormalDist normal(mu, sigma);
    return normal(*random);
}

double distrib_poisson(Random* random, double lambda)
{
    Log(Logger::LOG_DEBUG) << "distrib_poisson("
        << static_cast<void*>(random)
        << ", " << lambda << ")";

    PoissonDist poisson(lambda);
    return (double)poisson(*random);
}

double distrib_exponential(Random* random, double lambda)
{
    Log(Logger::LOG_DEBUG) << "distrib_exponential("
        << static_cast<void*>(random)
        << ", " << lambda << ")";

    ExponentialDist exponential(lambda);
    return exponential(*random);
}

double distrib_lognormal(Random* random, double mu, double sigma)
{
    Log(Logger::LOG_DEBUG) << "distrib_lognormal("
        << static_cast<void*>(random)
        << ", " << mu << ", " << sigma << ")";

    LognormalDist lognormal(mu, sigma);
    return lognormal(*random);
}

Random::~Random()
{
    --randomCount;
    Log(Logger::LOG_TRACE) << "deleted Random object, count: " << randomCount;
}

double Random::operator ()()
{
    double range = engine.max() - engine.min();
    return engine() / range;
}

void Random::setRandomSeed(int64_t val)
{
#ifdef RR_32BIT
    unsigned long maxl = std::numeric_limits<unsigned long>::max() - 2;
    unsigned long seed = val % maxl;
    engine.seed(seed);
    randomSeed = seed;
#else
    engine.seed(val);
    randomSeed = val;
#endif
}

int64_t Random::getRandomSeed()
{
    return randomSeed;
}

} /* namespace rrllvm */

