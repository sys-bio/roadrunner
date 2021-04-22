#ifndef ROADRUNNER_TESTMODELFACTORY
#define ROADRUNNER_TESTMODELFACTORY

#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <memory>
#include "Variant.h"

#include "rr-libstruct/lsMatrix.h"

//using DoublePair = std::pair<double, double>; // old swig no handle using statements
typedef std::pair<double, double> DoublePair;

/**
 * Data structure for storing reference simulation results.
 * std::string: model species name
 * std::pair<double, double>: mapping between starting value and simulation result.
 */
typedef std::unordered_map<std::string, DoublePair> StringDoublePairMap;

/**
 * Stores component name to expected model value mapping
 */
typedef std::unordered_map<std::string, double> StringDoubleMap;


/**
 * A collection of ResultMap objects
 * for testing models from multiple starting states.
 */
//using MultiResultsMap = std::vector<ResultMap>;
typedef std::vector<StringDoubleMap> VectorStringDoubleMap;

/**
 * Abstract type to store sbml string
 */
class TestModel {
public:

    /**
     * Returns the sbml string for this model
     */
    virtual std::string str() = 0;

    /**
     * Returns the model name. This is used by TestModelFactory
     * for instantiating instances of SBMLTestModel
     */
    virtual std::string modelName() = 0;

};

/**
 * @brief base class interface to result types.
 * @details exists only for any polymorphism that may
 * be implemented in future.
 */
class Result {
};

/**
 * Interface for time series result.
 * Models that implement this interface
 * are models to be used in testing time integrators
 */
class TimeSeriesResult : public Result {
public:
    /**
     * @brief returns the correct results for a time series simulation.
     * @details These results are always computed using an independent simulator.
     * Options provided in the timeSeriesSettings determine time, tol, duration etc.
     */
    virtual ls::DoubleMatrix timeSeriesResult() = 0;

    virtual std::unordered_map<std::string, rr::Variant> timeSeriesSettings();

};

/**
 * @brief Interface for classes that compute the steady state
 * from single parameter set.
 */
class SteadyStateResult : public Result {
public:

    virtual StringDoubleMap steadyState() = 0;

    /**
 * @brief Settings map for steady state tasks.
 * @details classes that implement this interface do not necessarily
 * need to override this method, since they can use default settings.
 */
    virtual std::unordered_map<std::string, rr::Variant> steadyStateSettings();


};

/**
 * @brief Interface for classes that compute the steady state
 * from multiple parameter sets.
 * @details SteadyStateResult was a bit limiting in that we can
 * only encode a single steady state test result from a single
 * parameter set. This interface addresses this problem.
 */
class SteadyStateMultiStart : public Result {
public:

    virtual VectorStringDoubleMap steadyState() = 0;
};


/**
 * A -> B; k1
 * B -> A; k2
 * k1 = 0.5
 * k2 = 1.0
 * A = 10;
 * B = 1;
 */
class SimpleFlux : public TestModel, public TimeSeriesResult, public SteadyStateResult {
public:

    std::string str() override;

    std::string modelName() override;

    ls::DoubleMatrix timeSeriesResult() override;

    StringDoubleMap steadyState() override;

    std::unordered_map<std::string, rr::Variant> steadyStateSettings() override;

    std::unordered_map<std::string, rr::Variant> timeSeriesSettings() override;
};

/**
 * @brief This model is the same as SimpleFlux but the network has been reduced
 * using conservation analysis manually before encoding in sbml.
 *
 * @details The system we solve is:
 *      f(x) =
 *       dS1/dt = - kf*S1 + kb*S2
 *       dS2/dt = + kf*S1 - kb*S2
 * We use
 *  - kf = 0.1
 *  - kb = 0.01
 *  - S1 @t0 = 10
 *  - S2 @t0 = 1
 *  - TOTAL = S1 + S2
 *
 * The solution to f(x) = 0 is:
 *  - S1 = 1
 *  - S2 = 10
 *
 * To get to this solution, we need to do conservation
 * analysis. Otherwise the jacobian matrix is singular,
 * and cannot be inverted (to solve newton iteration).
 * For this, we note that S1 + S2 = TOTAL, where TOTAL
 * is constant. Therefore we can reduce this system of
 * 2 equations into 1 differential and 1 algebraic.
 *
 * The reduced system is:
 *      f(x) =
 *       Total = S1 + S2
 *       S2Conserved = (Total - S1)
 *       dS1/dt = - kf*S1 + kb*S2Conserved
 *
 * The equivalent roadrunner/tellurium code is:
 *
 *  def ss(s):
 *      m = te.loada(s)
 *      m.conservedMoietyAnalysis = True
 *      m.steadyState()
 *      print(m.getFloatingSpeciesConcentrationIds())
 *      print(m.getFloatingSpeciesConcentrations())
 *      print(m.getFullJacobian())
 *      print(m.getReducedJacobian())
 *      print(m.getReducedStoichiometryMatrix())
 *      print(m.getGlobalParameterIds())
 *      print(m.getGlobalParameterValues())
 *      return m
 *  r = ss("""
 *  model x
 *      S1 = 10;
 *      S20 = 1;
 *      Total = S1 + S20;
 *      S1 => ; - kf*S1 + kb*(Total - S1);
 *      S2 := Total - S1;
 *      kf = 0.1;
 *      kb = 0.01;
 *  end
 *  """)
 *  The expected output:
 *      ['[S1]']
 *      [1.]
 *              S1
 *      S1 [[ 0.11]]
 *
 *              S1
 *      S1 [[ 0.11]]
 *
 *            _J0
 *      S1 [[  -1]]
 *
 *      ['S20', 'Total', 'kf', 'kb', 'S2']
 *      [1.0e+00 1.1e+01 1.0e-01 1.0e-02 1.0e+01]
 */
class SimpleFluxManuallyReduced : public TestModel, public SteadyStateResult {
public:
    std::string str() override;

    std::string modelName() override;

    StringDoubleMap steadyState() override;

    std::unordered_map<std::string, rr::Variant> steadyStateSettings() override;

};

/**
 * In [33]:  def ss(s):
    ...:      m = te.loada(s)
    ...:      m.conservedMoietyAnalysis = False
    ...:      m.steadyState()
    ...:      print(m.getFloatingSpeciesConcentrations())
    ...:      print(m.getFloatingSpeciesConcentrationIds())
    ...:      print(m.getFullJacobian())
    ...:      return m
    ...:
    ...:  r = ss("""
    ...:  model x
    ...:      S1 = 0;
    ...:      S2 = 0;
    ...:      => S1; kin
    ...:      S1 => S2; kf*S1;
    ...:      S2 => ; S2*kout;
    ...:      kf = 0.1;
    ...:      kb = 0.01;
    ...:      kin = 1;
    ...:      kout = 0.1
    ...:  end
    ...:  """)
 */
class OpenLinearFlux : public TestModel, public SteadyStateResult, public TimeSeriesResult {
public:
    std::string str() override;

    StringDoubleMap steadyState() override;

    ls::DoubleMatrix timeSeriesResult() override;

    std::string modelName() override;

    std::unordered_map<std::string, rr::Variant> steadyStateSettings() override;

    std::unordered_map<std::string, rr::Variant> timeSeriesSettings() override;

};

/**
 * model 269 from the sbml test suite
 */
class Model269 : public TestModel, public TimeSeriesResult {
public:

    std::string str() override;

    std::string modelName() override;

    ls::DoubleMatrix timeSeriesResult() override;

    std::unordered_map<std::string, rr::Variant> timeSeriesSettings() override;

};

/**
 * model 28 from the sbml test suite
 */
class Model28 : public TestModel, public TimeSeriesResult {
public:

    std::string str() override;

    std::string modelName() override;

    ls::DoubleMatrix timeSeriesResult() override;

    std::unordered_map<std::string, rr::Variant> timeSeriesSettings() override;

};


/**
 * A model that uses "ceil" in the rate law
 */
class CeilInRateLaw : public TestModel, public TimeSeriesResult {
public:

    std::string str() override;

    std::string modelName() override;

    ls::DoubleMatrix timeSeriesResult() override;

    std::unordered_map<std::string, rr::Variant> timeSeriesSettings() override;

};

/**
 * A model that uses "Factorial" in the rate law
 */
class FactorialInRateLaw : public TestModel, public TimeSeriesResult {
public:

    std::string str() override;

    std::string modelName() override;

    ls::DoubleMatrix timeSeriesResult() override;

    std::unordered_map<std::string, rr::Variant> timeSeriesSettings() override;

};

/**
 * Model from the Venkatraman 2010 paper
 */
class Venkatraman2010 : public TestModel, public SteadyStateResult {
public:

    std::string str() override;

    std::string modelName() override;

    StringDoubleMap steadyState() override;

    std::unordered_map<std::string, rr::Variant> steadyStateSettings() override;

};

class Brown2004 : public TestModel, public SteadyStateResult {

public:

    std::string str() override;

    std::string modelName() override;

    StringDoubleMap steadyState() override;

    std::unordered_map<std::string, rr::Variant> steadyStateSettings() override;

};

TestModel *TestModelFactory(const std::string &modelName);

/**
 * @brief returns a list of available model names
 * @note we could be smarter about this and implement a factory with self registration
 * (or use observer pattern?) however manually keeping track of these is fine for now
 * @warning Remember, if you extend the TestModelFactory, also extend this list
 */
std::vector<std::string> listOfModelNames();



namespace privateSwigTests_ {
    // this section exists only to test the swig bindings
    // and make sure the typemaps are doing what they are supposed
    // to be. Users should completely ignore this
    //
    // These were originally built as simple examples of how to
    // use swig. However, given the steep learning curve that is
    // swig - instead of deleting these they are left as an example
    // to future developers (including my future self.
    //     - ciaran
    //

    DoublePair *_testDoublePair(double first, double second);

    std::unordered_map<double, double> *_testDoubleMap(double first, double second);

    std::unordered_map<std::string, rr::Variant> *_testVariantMap();

    rr::Variant *_testVariant();

    StringDoublePairMap _testResultMap();
}


#endif // ROADRUNNER_TESTMODELFACTORY






