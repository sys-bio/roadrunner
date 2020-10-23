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

typedef cxx11_ns::bernoulli_distribution BernoulliDist;

typedef cxx11_ns::binomial_distribution<long> BinomialDist;

typedef cxx11_ns::cauchy_distribution<double> CauchyDist;

typedef cxx11_ns::chi_squared_distribution<double> ChisquareDist;

typedef cxx11_ns::exponential_distribution<double> ExponentialDist;

typedef cxx11_ns::gamma_distribution<double> GammaDist;

typedef cxx11_ns::lognormal_distribution<double> LognormalDist;

typedef cxx11_ns::poisson_distribution<int> PoissonDist;

static int randomCount = 0;

/**
 * random uniform distribution
 */

static double distrib_uniform(Random *random, double _min, double _max);

static double distrib_normal(Random* random, double mu, double sigma);

static double distrib_normal_four(Random* random, double mu, double sigma, double _min, double _max);

static double distrib_bernoulli(Random* random, double prob);

static double distrib_binomial(Random* random, double nTrials, double probabilityOfSuccess);

static double distrib_binomial_four(Random* random, double nTrials, double probabilityOfSuccess, double _min, double _max);

static double distrib_cauchy(Random* random, double location, double scale);

static double distrib_cauchy_one(Random* random, double scale);

static double distrib_cauchy_four(Random* random, double location, double scale, double _min, double _max);

static double distrib_chisquare(Random* random, double degreesOfFreedom);

static double distrib_chisquare_three(Random* random, double degreesOfFreedom, double _min, double _max);

static double distrib_exponential(Random* random, double lambda);

static double distrib_exponential_three(Random* random, double lambda, double _min, double _max);

static double distrib_gamma(Random* random, double shape, double scale);

static double distrib_gamma_four(Random* random, double shape, double scale, double _min, double _max);

static double distrib_laplace(Random* random, double location, double scale);

static double distrib_laplace_one(Random* random, double scale);

static double distrib_laplace_four(Random* random, double location, double scale, double _min, double _max);

static double distrib_lognormal(Random* random, double mu, double sigma);

static double distrib_lognormal_four(Random* random, double mu, double sigma, double _min, double _max);

static double distrib_poisson(Random* random, double lambda);

static double distrib_poisson_three(Random* random, double lambda, double _min, double _max);

static double distrib_rayleigh(Random* random, double scale);

static double distrib_rayleigh_three(Random* random, double scale, double _min, double _max);

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
    : normalized_uniform_dist(0.0, 1.0)
    , mMaxTries(100000)
{
    addGlobalMappings(ctx);
    setRandomSeed(defaultSeed());
    randomCount++;
}

Random::Random(const Random& other)
    : normalized_uniform_dist(0.0, 1.0)
    , mMaxTries(other.mMaxTries)
{
    *this = other;
    setRandomSeed(defaultSeed());
    randomCount++;
}

Random::Random()
    : normalized_uniform_dist(0.0, 1.0)
    , mMaxTries(100000)
{
    setRandomSeed(defaultSeed());
    randomCount++;
}

Random& Random::operator =(const Random& rhs)
{
    engine = rhs.engine;
    mMaxTries = rhs.mMaxTries;
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


    // LLVM does not appear to have a true void ptr, so just use a pointer
    // to a byte, pointers are all the same size anyway.
    // used for the LLVMModelData::random which is not accessed by
    // generated llvm code anyway.
    // see also, llvm::StructType *ModelDataIRBuilder::createModelDataStructType(...)
    Type *voidPtrType = Type::getInt8PtrTy(context);

    Type* args_void_double_quadruple[] = { voidPtrType, double_type, double_type, double_type, double_type };

    Type* args_void_double_triple[] = { voidPtrType, double_type, double_type, double_type };

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
        createGlobalMappingFunction("rr_distrib_normal_four",
            FunctionType::get(double_type, args_void_double_quadruple, false), module),
            (void*)distrib_normal_four);
    
    executionEngine->addGlobalMapping(
        createGlobalMappingFunction("rr_distrib_binomial",
            FunctionType::get(double_type, args_void_double_double, false), module),
        (void*)distrib_binomial);

    executionEngine->addGlobalMapping(
        createGlobalMappingFunction("rr_distrib_bernoulli",
            FunctionType::get(double_type, args_void_double, false), module),
        (void*)distrib_bernoulli);

    executionEngine->addGlobalMapping(
        createGlobalMappingFunction("rr_distrib_binomial_four",
            FunctionType::get(double_type, args_void_double_quadruple, false), module),
        (void*)distrib_binomial_four);

    executionEngine->addGlobalMapping(
        createGlobalMappingFunction("rr_distrib_cauchy",
            FunctionType::get(double_type, args_void_double_double, false), module),
        (void*)distrib_cauchy);

    executionEngine->addGlobalMapping(
        createGlobalMappingFunction("rr_distrib_cauchy_one",
            FunctionType::get(double_type, args_void_double, false), module),
        (void*)distrib_cauchy_one);

    executionEngine->addGlobalMapping(
        createGlobalMappingFunction("rr_distrib_cauchy_four",
            FunctionType::get(double_type, args_void_double_quadruple, false), module),
        (void*)distrib_cauchy_four);

    executionEngine->addGlobalMapping(
        createGlobalMappingFunction("rr_distrib_chisquare",
            FunctionType::get(double_type, args_void_double, false), module),
        (void*)distrib_chisquare);

    executionEngine->addGlobalMapping(
        createGlobalMappingFunction("rr_distrib_chisquare_three",
            FunctionType::get(double_type, args_void_double_triple, false), module),
        (void*)distrib_chisquare_three);

    executionEngine->addGlobalMapping(
        createGlobalMappingFunction("rr_distrib_exponential",
            FunctionType::get(double_type, args_void_double, false), module),
            (void*)distrib_exponential);

    executionEngine->addGlobalMapping(
        createGlobalMappingFunction("rr_distrib_exponential_three",
            FunctionType::get(double_type, args_void_double_triple, false), module),
            (void*)distrib_exponential_three);

    executionEngine->addGlobalMapping(
        createGlobalMappingFunction("rr_distrib_gamma",
            FunctionType::get(double_type, args_void_double_double, false), module),
        (void*)distrib_gamma);

    executionEngine->addGlobalMapping(
        createGlobalMappingFunction("rr_distrib_gamma_four",
            FunctionType::get(double_type, args_void_double_quadruple, false), module),
        (void*)distrib_gamma_four);

    executionEngine->addGlobalMapping(
        createGlobalMappingFunction("rr_distrib_laplace",
            FunctionType::get(double_type, args_void_double_double, false), module),
        (void*)distrib_laplace);

    executionEngine->addGlobalMapping(
        createGlobalMappingFunction("rr_distrib_laplace_one",
            FunctionType::get(double_type, args_void_double, false), module),
        (void*)distrib_laplace_one);

    executionEngine->addGlobalMapping(
        createGlobalMappingFunction("rr_distrib_laplace_four",
            FunctionType::get(double_type, args_void_double_quadruple, false), module),
        (void*)distrib_laplace_four);

    executionEngine->addGlobalMapping(
        createGlobalMappingFunction("rr_distrib_lognormal",
            FunctionType::get(double_type, args_void_double_double, false), module),
            (void*)distrib_lognormal);

    executionEngine->addGlobalMapping(
        createGlobalMappingFunction("rr_distrib_lognormal_four",
            FunctionType::get(double_type, args_void_double_quadruple, false), module),
            (void*)distrib_lognormal_four);

    executionEngine->addGlobalMapping(
        createGlobalMappingFunction("rr_distrib_poisson",
            FunctionType::get(double_type, args_void_double, false), module),
        (void*)distrib_poisson);

    executionEngine->addGlobalMapping(
        createGlobalMappingFunction("rr_distrib_poisson_three",
            FunctionType::get(double_type, args_void_double_triple, false), module),
        (void*)distrib_poisson_three);

    executionEngine->addGlobalMapping(
        createGlobalMappingFunction("rr_distrib_rayleigh",
            FunctionType::get(double_type, args_void_double, false), module),
        (void*)distrib_rayleigh);

    executionEngine->addGlobalMapping(
        createGlobalMappingFunction("rr_distrib_rayleigh_three",
            FunctionType::get(double_type, args_void_double_triple, false), module),
        (void*)distrib_rayleigh_three);

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
    return dist(random->engine);
}

double distrib_normal(Random* random, double mu, double sigma)
{
    Log(Logger::LOG_DEBUG) << "distrib_normal("
            << static_cast<void*>(random)
            << ", " << mu << ", " << sigma << ")";

    NormalDist normal(mu, sigma);
    return normal(random->engine);
}

double distrib_normal_four(Random* random, double mu, double sigma, double _min, double _max)
{
    Log(Logger::LOG_DEBUG) << "distrib_normal("
        << static_cast<void*>(random)
        << ", " << mu << ", " << sigma << ", " << _min << ", " << _max << ")";

    if (_min > _max) {
        Log(Logger::LOG_DEBUG) << "Invalid call to function: " << _min << " is greater than " << _max << ".";
        return nan("");
    }
    if (_min == _max) {
        return _min;
    }

    NormalDist normal(mu, sigma);
    double distval = normal(random->engine);
    int count = 0;
    while (count < random->getMaxTries() && (distval<_min || distval >= _max)) {
        distval = normal(random->engine);
        count++;
    }
    if (count == random->getMaxTries()) {
        Log(Logger::LOG_DEBUG) << "Unable to draw from truncated distribution after " << count << " tries.  Using the midpoint between " << _min << " and " << _max << " instead.";
        distval = (_min + _max)/2;
    }
    
    return distval;
}

double distrib_bernoulli(Random* random, double prob)
{
    Log(Logger::LOG_DEBUG) << "distrib_bernoulli("
        << static_cast<void*>(random)
        << ", " << prob << ")";

    BernoulliDist bernoulli(prob);
    return bernoulli(random->engine);
}

double distrib_binomial(Random* random, double nTrials, double probabilityOfSuccess)
{
    Log(Logger::LOG_DEBUG) << "distrib_binomial("
        << static_cast<void*>(random)
        << ", " << nTrials << ", " << probabilityOfSuccess << ")";

    BinomialDist binomial(roundl(nTrials), probabilityOfSuccess);
    return binomial(random->engine);
}

double distrib_binomial_four(Random* random, double nTrials, double probabilityOfSuccess, double _min, double _max)
{
    Log(Logger::LOG_DEBUG) << "distrib_binomial("
        << static_cast<void*>(random)
        << ", " << nTrials << ", " << probabilityOfSuccess << ", " << _min << ", " << _max << ")";

    if (_min > _max) {
        Log(Logger::LOG_DEBUG) << "Invalid call to function: " << _min << " is greater than " << _max << ".";
        return nan("");
    }
    if (_min == _max) {
        return _min;
    }

    BinomialDist binomial(roundl(nTrials), probabilityOfSuccess);
    long distval = binomial(random->engine);
    int count = 0;
    while (count < random->getMaxTries() && (distval < _min || distval > _max)) {
        distval = binomial(random->engine);
        count++;
    }
    if (count == random->getMaxTries()) {
        Log(Logger::LOG_DEBUG) << "Unable to draw from truncated distribution after " << count << " tries.  Using the midpoint between " << _min << " and " << _max << " instead.";
        distval = (std::max(0.0, _min) + std::min(nTrials, _max))/2;
    }

    return distval;
}

double distrib_cauchy(Random* random, double location, double scale)
{
    Log(Logger::LOG_DEBUG) << "distrib_cauchy("
        << static_cast<void*>(random)
        << ", " << location << ", " << scale << ")";

    CauchyDist cauchy(location, scale);
    return cauchy(random->engine);
}

double distrib_cauchy_one(Random* random, double scale)
{
    Log(Logger::LOG_DEBUG) << "distrib_cauchy("
        << static_cast<void*>(random)
        << ", " << scale << ")";

    //If only 'scale' is provided, use 0.0 as the 'location'.
    CauchyDist cauchy(0.0, scale);
    return cauchy(random->engine);
}

double distrib_cauchy_four(Random* random, double location, double scale, double _min, double _max)
{
    Log(Logger::LOG_DEBUG) << "distrib_cauchy("
        << static_cast<void*>(random)
        << ", " << location << ", " << scale << ", " << _min << ", " << _max << ")";

    if (_min > _max) {
        Log(Logger::LOG_DEBUG) << "Invalid call to function: " << _min << " is greater than " << _max << ".";
        return nan("");
    }
    if (_min == _max) {
        return _min;
    }

    CauchyDist cauchy(location, scale);
    double distval = cauchy(random->engine);
    int count = 0;
    while (count < random->getMaxTries() && (distval < _min || distval >= _max)) {
        distval = cauchy(random->engine);
        count++;
    }
    if (count == random->getMaxTries()) {
        Log(Logger::LOG_DEBUG) << "Unable to draw from truncated distribution after " << count << " tries.  Using the midpoint between " << _min << " and " << _max << " instead.";
        distval = (_min + _max)/2;
    }

    return distval;
}

double distrib_chisquare(Random* random, double degreesOfFreedom)
{
    Log(Logger::LOG_DEBUG) << "distrib_chisquare("
        << static_cast<void*>(random)
        << ", " << degreesOfFreedom << ")";

    ChisquareDist chisquare(degreesOfFreedom);
    return (double)chisquare(random->engine);
}

double distrib_chisquare_three(Random* random, double degreesOfFreedom, double _min, double _max)
{
    Log(Logger::LOG_DEBUG) << "distrib_chisquare("
        << static_cast<void*>(random)
        << ", " << degreesOfFreedom << ", " << _min << ", " << _max << ")";

    if (_min > _max) {
        Log(Logger::LOG_DEBUG) << "Invalid call to function: " << _min << " is greater than " << _max << ".";
        return nan("");
    }
    if (_min == _max) {
        return _min;
    }

    ChisquareDist chisquare(degreesOfFreedom);
    double distval = (double)chisquare(random->engine);
    int count = 0;
    while (count < random->getMaxTries() && (distval<_min || distval > _max)) {
        distval = (double)chisquare(random->engine);
        count++;
    }
    if (count == random->getMaxTries()) {
        Log(Logger::LOG_DEBUG) << "Unable to draw from truncated distribution after " << count << " tries.  Using the midpoint between " << _min << " and " << _max << " instead.";
        distval = (_min + _max)/2;
    }

    return distval;
}

double distrib_exponential(Random* random, double lambda)
{
    Log(Logger::LOG_DEBUG) << "distrib_exponential("
        << static_cast<void*>(random)
        << ", " << lambda << ")";

    ExponentialDist exponential(lambda);
    return exponential(random->engine);
}

double distrib_exponential_three(Random* random, double lambda, double _min, double _max)
{
    Log(Logger::LOG_DEBUG) << "distrib_exponential("
        << static_cast<void*>(random)
        << ", " << lambda << ", " << _min << ", " << _max << ")";

    if (_min > _max) {
        Log(Logger::LOG_DEBUG) << "Invalid call to function: " << _min << " is greater than " << _max << ".";
        return nan("");
    }
    if (_min == _max) {
        return _min;
    }

    ExponentialDist exponential(lambda);
    double distval = exponential(random->engine);
    int count = 0;
    while (count < random->getMaxTries() && (distval<_min || distval >= _max)) {
        distval = exponential(random->engine);
        count++;
    }
    if (count == random->getMaxTries()) {
        Log(Logger::LOG_DEBUG) << "Unable to draw from truncated distribution after " << count << " tries.  Using the midpoint between " << _min << " and " << _max << " instead.";
        distval = (_min + _max)/2;
    }

    return distval;
}

double distrib_gamma(Random* random, double shape, double scale)
{
    Log(Logger::LOG_DEBUG) << "distrib_gamma("
        << static_cast<void*>(random)
        << ", " << shape << ", " << scale << ")";

    GammaDist gamma(shape, scale);
    return gamma(random->engine);
}

double distrib_gamma_four(Random* random, double shape, double scale, double _min, double _max)
{
    Log(Logger::LOG_DEBUG) << "distrib_gamma("
        << static_cast<void*>(random)
        << ", " << shape << ", " << scale << ", " << _min << ", " << _max << ")";

    if (_min > _max) {
        Log(Logger::LOG_DEBUG) << "Invalid call to function: " << _min << " is greater than " << _max << ".";
        return nan("");
    }
    if (_min == _max) {
        return _min;
    }

    GammaDist gamma(shape, scale);
    double distval = gamma(random->engine);
    int count = 0;
    while (count < random->getMaxTries() && (distval < _min || distval >= _max)) {
        distval = gamma(random->engine);
        count++;
    }
    if (count == random->getMaxTries()) {
        Log(Logger::LOG_DEBUG) << "Unable to draw from truncated distribution after " << count << " tries.  Using the midpoint between " << _min << " and " << _max << " instead.";
        distval = (_min + _max)/2;
    }

    return distval;
}

double distrib_laplace(Random* random, double location, double scale)
{
    Log(Logger::LOG_DEBUG) << "distrib_laplace("
        << static_cast<void*>(random)
        << ", " << location << ", " << scale << ")";

    ExponentialDist laplace(1/scale);
    double exp1 = laplace(random->engine);
    double exp2 = laplace(random->engine);
    return exp1 - exp2 + location;
}

double distrib_laplace_one(Random* random, double scale)
{
    Log(Logger::LOG_DEBUG) << "distrib_laplace("
        << static_cast<void*>(random)
        << ", " << scale << ")";

    //If only 'scale' is provided, use 0.0 as the 'location'.
    ExponentialDist laplace(1/scale);
    double exp1 = laplace(random->engine);
    double exp2 = laplace(random->engine);
    return exp1 - exp2;
}

double distrib_laplace_four(Random* random, double location, double scale, double _min, double _max)
{
    Log(Logger::LOG_DEBUG) << "distrib_laplace("
        << static_cast<void*>(random)
        << ", " << location << ", " << scale << ", " << _min << ", " << _max << ")";

    if (_min > _max) {
        Log(Logger::LOG_DEBUG) << "Invalid call to function: " << _min << " is greater than " << _max << ".";
        return nan("");
    }
    if (_min == _max) {
        return _min;
    }

    ExponentialDist laplace(1/scale);
    double exp1 = laplace(random->engine);
    double exp2 = laplace(random->engine);
    double distval = exp1 - exp2 + location;
    int count = 0;
    while (count < random->getMaxTries() && (distval < _min || distval >= _max)) {
        exp1 = laplace(random->engine);
        exp2 = laplace(random->engine);
        distval = exp1 - exp2 + location;
        count++;
    }
    if (count == random->getMaxTries()) {
        Log(Logger::LOG_DEBUG) << "Unable to draw from truncated distribution after " << count << " tries.  Using the midpoint between " << _min << " and " << _max << " instead.";
        distval = (_min + _max)/2;
    }

    return distval;
}

double distrib_lognormal(Random* random, double mu, double sigma)
{
    Log(Logger::LOG_DEBUG) << "distrib_lognormal("
        << static_cast<void*>(random)
        << ", " << mu << ", " << sigma << ")";

    LognormalDist lognormal(mu, sigma);
    return lognormal(random->engine);
}

double distrib_lognormal_four(Random* random, double mu, double sigma, double _min, double _max)
{
    Log(Logger::LOG_DEBUG) << "distrib_lognormal("
        << static_cast<void*>(random)
        << ", " << mu << ", " << sigma << ", " << _min << ", " << _max << ")";

    if (_min > _max) {
        Log(Logger::LOG_DEBUG) << "Invalid call to function: " << _min << " is greater than " << _max << ".";
        return nan("");
    }
    if (_min == _max) {
        return _min;
    }

    LognormalDist lognormal(mu, sigma);
    double distval = lognormal(random->engine);
    int count = 0;
    while (count < random->getMaxTries() && (distval < _min || distval >= _max)) {
        distval = lognormal(random->engine);
        count++;
    }
    if (count == random->getMaxTries()) {
        Log(Logger::LOG_DEBUG) << "Unable to draw from truncated distribution after " << count << " tries.  Using the midpoint between " << _min << " and " << _max << " instead.";
        distval = (_min + _max)/2;
    }

    return distval;
}

double distrib_poisson(Random* random, double lambda)
{
    Log(Logger::LOG_DEBUG) << "distrib_poisson("
        << static_cast<void*>(random)
        << ", " << lambda << ")";

    PoissonDist poisson(lambda);
    return (double)poisson(random->engine);
}

double distrib_poisson_three(Random* random, double lambda, double _min, double _max)
{
    Log(Logger::LOG_DEBUG) << "distrib_poisson("
        << static_cast<void*>(random)
        << ", " << lambda << ", " << _min << ", " << _max << ")";

    if (_min > _max) {
        Log(Logger::LOG_DEBUG) << "Invalid call to function: " << _min << " is greater than " << _max << ".";
        return nan("");
    }
    if (_min == _max) {
        return _min;
    }

    PoissonDist poisson(lambda);
    double distval = (double)poisson(random->engine);
    int count = 0;
    while (count < random->getMaxTries() && (distval<_min || distval > _max)) {
        distval = (double)poisson(random->engine);
        count++;
    }
    if (count == random->getMaxTries()) {
        Log(Logger::LOG_DEBUG) << "Unable to draw from truncated distribution after " << count << " tries.  Using the midpoint between " << _min << " and " << _max << " instead.";
        distval = (_min + _max)/2;
    }

    return distval;
}

double distrib_rayleigh(Random* random, double scale)
{
    Log(Logger::LOG_DEBUG) << "distrib_rayleigh("
        << static_cast<void*>(random)
        << ", " << scale << ")";

#ifdef CXX11_RANDOM
    cxx11_ns::uniform_real_distribution<double> dist(0, 1);
#else
    cxx11_ns::uniform_real<double> dist(0, 1);
#endif
    return scale * sqrt(-2*log(dist(random->engine)));

}

double distrib_rayleigh_three(Random* random, double scale, double _min, double _max)
{
    Log(Logger::LOG_DEBUG) << "distrib_rayleigh("
        << static_cast<void*>(random)
        << ", " << scale << ", " << _min << ", " << _max << ")";

    if (_min > _max) {
        Log(Logger::LOG_DEBUG) << "Invalid call to function: " << _min << " is greater than " << _max << ".";
        return nan("");
    }
    if (_min == _max) {
        return _min;
    }

#ifdef CXX11_RANDOM
    cxx11_ns::uniform_real_distribution<double> dist(0, 1);
#else
    cxx11_ns::uniform_real<double> dist(0, 1);
#endif
    double distval = scale * sqrt(-2 * log(dist(random->engine)));
    int count = 0;
    while (count < random->getMaxTries() && (distval<_min || distval >= _max)) {
        distval = scale * sqrt(-2 * log(dist(random->engine)));
        count++;
    }
    if (count == random->getMaxTries()) {
        Log(Logger::LOG_DEBUG) << "Unable to draw from truncated distribution after " << count << " tries.  Using the midpoint between " << _min << " and " << _max << " instead.";
        distval = (_min + _max)/2;
    }

    return distval;
}

Random::~Random()
{
    --randomCount;
    Log(Logger::LOG_TRACE) << "deleted Random object, count: " << randomCount;
}

double Random::operator ()()
{
	// uniform_real is better than directly scaling engine() output because engine
	// is intended as an unsigned random bit generator, not a real number generator
	return normalized_uniform_dist(engine);
}

int Random::getMaxTries() const
{
    return mMaxTries;
}

void Random::setMaxTries(int maxTries)
{
    mMaxTries = maxTries;
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

