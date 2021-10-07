#ifndef ROADRUNNER_TESTMODELFACTORY
#define ROADRUNNER_TESTMODELFACTORY

#include <vector>

std::vector<int> getVofI();

#include <unordered_map>
//#include <stdexcept>
//#include <memory>
#include "Setting.h"
#include "rrRoadRunner.h"
#include "Matrix.h"
#include "Matrix3D.h"
//#include <complex>
//
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

    /**
     * @brief write sbml to file
     */
    void toFile(const std::string &fname);

    virtual ~TestModel() = default;

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
    virtual rr::Matrix<double> timeSeriesResult() = 0;

    virtual std::unordered_map<std::string, rr::Setting> timeSeriesSettings();

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
    virtual std::unordered_map<std::string, rr::Setting> steadyStateSettings();

    /**
     * @brief Apply whatever settings from steadyStateSettings to the model
     * pointed to by @param *rr
     */
    void applySteadyStateSettings(rr::RoadRunner *rr);

    /**
     * @brief Compare steady state of roadrunner model
     * pointed to by @param rr with those in
     * SteadyStateResult::steadyState
     * @details settings are first applied from
     * steadyStateSettings using applySteadyStateSettings. Then a
     * call to RoadRunner::steadyState computes the steady state
     * and makes the comparison between expected values and actual
     * computed values.
     */
    void checkSteadyState(rr::RoadRunner *rr, double tol = 1e-4);

};

/**
 * @brief TestModel classes that implement this interface
 * test steady state fluxes.
 * @details Uses steadyStateSettings defined in SteadyStateResult
 */
class SteadyStateFluxes : public SteadyStateResult {
public:

    virtual std::unordered_map<std::string, double> steadyStateFluxes() = 0;

};

/**
 * @brief Interface for classes that compute the steady state
 * from multiple parameter sets.
 * @details SteadyStateResult was a bit limiting in that we can
 * only encode a single steady state test result from a single
 * parameter set. This interface addresses this problem.
 * @note this has not been used
 */
class SteadyStateMultiStart : public Result {
public:

    virtual VectorStringDoubleMap steadyState() = 0;
};

/**
 * @brief interface for models that test the full jacobian matrix.
 * @details classes that implement this interface
 * are expected to hard code the results of the
 * jacobian matrix, calculated with an independent
 * tool (or better - analytically ) at the settings defined by
 * jacobianSettings. Models that implement this interface
 * only test the full jacobian. If a model has conserved
 * cycles it should instead implement the JacobianResultReduced
 * interface which have been split (according to the interface
 * segregation principle).
 */
class JacobianResultFull : public Result {
public:
    virtual rr::Matrix<double> fullJacobianAmt() = 0;

    virtual rr::Matrix<double> fullJacobianConc() = 0;

    virtual std::unordered_map<std::string, rr::Setting> jacobianSettings() = 0;
};

/**
 * @brief Test the reduced jacobian.
 * @details TestModel types that implement the JacobianResultReduced
 * interface have declared that they not only test the jacobian matrix,
 * but that the model has conserved cycles and therefore the
 * full and reduced jacobians are different. Since this
 * interface inherits from JacobianResultFull, TestModel types
 * that implement this interface must also implement the
 * JacoianResultFull interface.
 * Since we are testing the reduced jacobian, it is implied
 * that moiety conservation option in RoadRunner is
 * to be turned on.
 * The jacobianSettings from JacobianResultFull is reused
 * and should indicate the conditions where the TestModel jacobian
 * reference data and computed jacobian should match. For instance,
 * set "steady_state" to true or time to an appropriate time point.
 */
class JacobianResultReduced : public JacobianResultFull {
public:

    virtual rr::Matrix<double> reducedJacobianAmt() = 0;

    virtual rr::Matrix<double> reducedJacobianConc() = 0;

};

/**
 * @brief interface for testing the eigenvalues of a model
 */
class EigenResult : public Result {
public:

    virtual std::vector<ls::Complex> fullEigenValues() = 0;

    virtual std::vector<ls::Complex> reducedEigenValues() = 0;

    virtual std::unordered_map<std::string, rr::Setting> eigenSettings() = 0;

};

/**
 * @brief TestModels that implement this interface
 * can test structural properties of a model
 */
class StructuralProperties : public Result {
public:

    virtual rr::Matrix<double> linkMatrix() = 0;

    virtual rr::Matrix<double> NrMatrix() = 0;

    virtual rr::Matrix<double> KMatrix() = 0;

    virtual rr::Matrix<double> reducedStoicMatrix() = 0;

    virtual rr::Matrix<double> fullStoicMatrix() = 0;

    virtual rr::Matrix<double> extendedStoicMatrix() = 0;

    virtual rr::Matrix<double> L0Matrix() = 0;

    virtual rr::Matrix<double> conservationMatrix() = 0;
};

/**
 * @brief TestModel class that implement this interface
 * can test metabolic control analysis
 */
class MCAResult : public Result {
public:
    virtual rr::Matrix<double> unscaledConcentrationControlCoefficientMatrix() = 0;

    virtual rr::Matrix<double> scaledConcentrationControlCoefficientMatrix() = 0;

    virtual rr::Matrix<double> unscaledFluxControlCoefficientMatrix() = 0;

    virtual rr::Matrix<double> scaledFluxControlCoefficientMatrix() = 0;

//    virtual rr::Matrix<double> unscaledParameterElasticity() = 0;

    virtual rr::Matrix<double> unscaledElasticityMatrix() = 0;

    virtual rr::Matrix<double> scaledElasticityMatrix() = 0;

    virtual std::unordered_map<std::string, rr::Setting> mcaSettings() = 0;
};


class SensitivityResult : public Result {
};

class TimeSeriesSensitivityResult : public SensitivityResult {
public:
    virtual std::unordered_map<std::string, rr::Setting> timeSeriesSensitivityResultSettings() = 0;

    virtual rr::Matrix3D<double, double> timeSeriesSensitivityResult() = 0;

};

class SteadyStateSensitivityResult : public SensitivityResult {
public:
    virtual std::unordered_map<std::string, rr::Setting> steadyStateSensitivityResultSettings() = 0;

    virtual rr::Matrix<double> steadyStateSensitivityResult() = 0;

};


/**
 * A -> B; k1
 * B -> A; k2
 * k1 = 0.5
 * k2 = 1.0
 * A = 10;
 * B = 1;
 */
class SimpleFlux :
        public TestModel,
        public TimeSeriesResult,
        public SteadyStateResult,
        public JacobianResultReduced,
        public EigenResult,
        public StructuralProperties,
        public MCAResult,
        public TimeSeriesSensitivityResult {
public:

    std::string str() override;

    std::string modelName() override;

    rr::Matrix<double> timeSeriesResult() override;

    StringDoubleMap steadyState() override;

    rr::Matrix<double> fullJacobianAmt() override;

    rr::Matrix<double> fullJacobianConc() override;

    rr::Matrix<double> reducedJacobianAmt() override;

    rr::Matrix<double> reducedJacobianConc() override;

    std::vector<ls::Complex> fullEigenValues() override;

    std::vector<ls::Complex> reducedEigenValues() override;

    std::unordered_map<std::string, rr::Setting> eigenSettings() override;

    std::unordered_map<std::string, rr::Setting> jacobianSettings() override;

    std::unordered_map<std::string, rr::Setting> steadyStateSettings() override;

    std::unordered_map<std::string, rr::Setting> timeSeriesSettings() override;

    rr::Matrix<double> linkMatrix() override;

    rr::Matrix<double> NrMatrix() override;

    rr::Matrix<double> KMatrix() override;

    rr::Matrix<double> reducedStoicMatrix() override;

    rr::Matrix<double> fullStoicMatrix() override;

    rr::Matrix<double> extendedStoicMatrix() override;

    rr::Matrix<double> L0Matrix() override;

    rr::Matrix<double> conservationMatrix() override;

    rr::Matrix<double> unscaledConcentrationControlCoefficientMatrix() override;

    rr::Matrix<double> scaledConcentrationControlCoefficientMatrix() override;

    rr::Matrix<double> unscaledFluxControlCoefficientMatrix() override;

    rr::Matrix<double> scaledFluxControlCoefficientMatrix() override;

//    rr::Matrix<double> unscaledParameterElasticity() override;

    rr::Matrix<double> unscaledElasticityMatrix() override;

    rr::Matrix<double> scaledElasticityMatrix() override;

    std::unordered_map<std::string, rr::Setting> mcaSettings() override;

    std::unordered_map<std::string, rr::Setting> timeSeriesSensitivityResultSettings() override;

    rr::Matrix3D<double, double> timeSeriesSensitivityResult() override;
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

    std::unordered_map<std::string, rr::Setting> steadyStateSettings() override;

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
    ...:  model OpenLinearFlux
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
class OpenLinearFlux :
        public TestModel,
        public SteadyStateResult,
        public TimeSeriesResult,
        public JacobianResultFull,
        public TimeSeriesSensitivityResult {
public:
    std::string str() override;

    StringDoubleMap steadyState() override;

    rr::Matrix<double> timeSeriesResult() override;

    std::string modelName() override;

    std::unordered_map<std::string, rr::Setting> steadyStateSettings() override;

    std::unordered_map<std::string, rr::Setting> timeSeriesSettings() override;

    rr::Matrix<double> fullJacobianAmt() override;

    rr::Matrix<double> fullJacobianConc() override;

    std::unordered_map<std::string, rr::Setting> jacobianSettings() override;

    std::unordered_map<std::string, rr::Setting> timeSeriesSensitivityResultSettings() override;

    rr::Matrix3D<double, double> timeSeriesSensitivityResult() override;

};

/**
 * model 269 from the sbml test suite
 */
class Model269 : public TestModel, public TimeSeriesResult, public TimeSeriesSensitivityResult {
public:

    std::string str() override;

    std::string modelName() override;

    rr::Matrix<double> timeSeriesResult() override;

    std::unordered_map<std::string, rr::Setting> timeSeriesSettings() override;

    std::unordered_map<std::string, rr::Setting> timeSeriesSensitivityResultSettings() override;

    rr::Matrix3D<double, double> timeSeriesSensitivityResult() override;
};

/**
 * model 28 from the sbml test suite
 */
class Model28 : public TestModel, public TimeSeriesResult, public TimeSeriesSensitivityResult {
public:

    std::string str() override;

    std::string modelName() override;

    rr::Matrix<double> timeSeriesResult() override;

    std::unordered_map<std::string, rr::Setting> timeSeriesSettings() override;

    rr::Matrix3D<double, double> timeSeriesSensitivityResult() override;

    std::unordered_map<std::string, rr::Setting> timeSeriesSensitivityResultSettings() override;
};


/**
 * A model that uses "ceil" in the rate law
 */
class CeilInRateLaw : public TestModel, public TimeSeriesResult {
public:

    std::string str() override;

    std::string modelName() override;

    rr::Matrix<double> timeSeriesResult() override;

    std::unordered_map<std::string, rr::Setting> timeSeriesSettings() override;

};

/**
 * A model that uses "Factorial" in the rate law
 */
class FactorialInRateLaw : public TestModel, public TimeSeriesResult {
public:

    std::string str() override;

    std::string modelName() override;

    rr::Matrix<double> timeSeriesResult() override;

    std::unordered_map<std::string, rr::Setting> timeSeriesSettings() override;

};

/**
 * Model from the Venkatraman 2010 paper
 */
class Venkatraman2010 : public TestModel, public SteadyStateResult, public JacobianResultFull {
public:

    std::string str() override;

    std::string modelName() override;

    StringDoubleMap steadyState() override;

    std::unordered_map<std::string, rr::Setting> steadyStateSettings() override;

    rr::Matrix<double> fullJacobianAmt() override;

    rr::Matrix<double> fullJacobianConc() override;

    std::unordered_map<std::string, rr::Setting> jacobianSettings() override;

};

class Brown2004 :
        public TestModel,
        public SteadyStateResult
//        public TimeSeriesResult
{

public:

    std::string str() override;

    std::string modelName() override;

    StringDoubleMap steadyState() override;

    std::unordered_map<std::string, rr::Setting> steadyStateSettings() override;

//    rr::Matrix<double> timeSeriesResult() override;
//
//    std::unordered_map<std::string, rr::Setting> timeSeriesSettings() override;

};


/**
 * This SBML model does not have any rate laws not parameters.
 * It contains layout information only, which is valid sbml.
 * This model comes from issue-756
 *  https://github.com/sys-bio/roadrunner/issues/756
 */
class LayoutOnly : public TestModel {
public:

    std::string str() override;

    std::string modelName() override;
};


class ModelWithLocalParameters : public TestModel {
public:
    std::string str() override;

    std::string modelName() override;
};

/**
 * @brief
 * @details This test model was originally defined inside Biomolecular_end.rrtest
 *  (reimplemented here for debugging a problem that is hard to debug with original test)
 */
class BimolecularEnd :
        public TestModel,
        public SteadyStateFluxes,
        public JacobianResultFull,
        public StructuralProperties,
        public MCAResult {
public:
    std::string modelName() override;

    std::string str() override;

    std::unordered_map<std::string, double> steadyState() override;

    std::unordered_map<std::string, rr::Setting> steadyStateSettings() override;

    std::unordered_map<std::string, double> steadyStateFluxes() override;

    std::unordered_map<std::string, rr::Setting> jacobianSettings() override;

    rr::Matrix<double> fullJacobianConc() override;

    /**
     * Amt is same as conc because volume of single compartment == 1
     */
    rr::Matrix<double> fullJacobianAmt() override;

    rr::Matrix<double> linkMatrix() override;

    rr::Matrix<double> NrMatrix() override;

    rr::Matrix<double> KMatrix() override;

    rr::Matrix<double> reducedStoicMatrix() override;

    rr::Matrix<double> fullStoicMatrix() override;

    rr::Matrix<double> extendedStoicMatrix() override;

    rr::Matrix<double> L0Matrix() override;

    rr::Matrix<double> conservationMatrix() override;

    rr::Matrix<double> unscaledConcentrationControlCoefficientMatrix() override;

    rr::Matrix<double> scaledConcentrationControlCoefficientMatrix() override;

    rr::Matrix<double> unscaledFluxControlCoefficientMatrix() override;

    rr::Matrix<double> scaledFluxControlCoefficientMatrix() override;

    rr::Matrix<double> unscaledElasticityMatrix() override;

    rr::Matrix<double> scaledElasticityMatrix() override;

    std::unordered_map<std::string, rr::Setting> mcaSettings() override;
};

/**
 * @brief this is model number 00039 from the stochastic sbml test suite.
 */
class BatchImmigrationDeath03 : public TestModel {
public:

    std::string str() override;

    std::string modelName() override;
};


/**
 * @brief returns a vector of strings that are the
 * names of the classes that inherit from TestModel.
 */
std::vector<std::string> getAvailableTestModels();

/**
 * @brief Create instances of TestModel
 * @param modelName the name of the model to create as a string
 * @see getAvailableTestModels for list of available models.
 */
TestModel *TestModelFactory(const std::string &modelName);


namespace privateSwigTests_ {
    // this section exists only to test the swig bindings
    // and make sure the typemaps are doing what they are supposed
    // to. Users should never see this
    //
    // These were originally built as simple examples of how to
    // use swig. However, given the steep learning curve that is
    // swig, instead of deleting they are left as an example
    // to future developers (including my future self).
    //

    DoublePair *_testDoublePair(double first, double second);

    std::unordered_map<double, double> *_testDoubleMap(double first, double second);

    std::unordered_map<std::string, rr::Setting> *_testVariantMap();

    rr::Setting *_testVariant();

    StringDoublePairMap _testResultMap();

    std::vector<std::complex<double>> _testStdComplexZeroImagPart();

    std::vector<std::complex<double>> _testStdComplexNonZeroImagPart();

    std::vector<double> _testDoubleVectorTo1DNumpyArray();

    rr::Matrix3D<double, double> _testMatrix3D_3x2x3();

    rr::Matrix3D<double, double> _testMatrix3D_2x3x4();

    rr::Matrix3D<double, double> _testMatrix3D_4x3x2();

    rr::Matrix3D<double, double> _testMatrix3D_3x4x2();

    ls::Matrix<double> _testLsMatrixWithLabels();

    rr::Matrix<double> _testRRMatrixWithLabels();

    std::string _testPythonStringToCxxRoundTrip(std::string s);

    std::vector<std::string> _testAddElementToStringVec(std::vector<std::string> stringVec, std::string newElement);

    std::vector<std::string>& _testAddElementToStringVecAsReference( std::vector< std::string>& stringVec, std::string newElement);

}


#endif // ROADRUNNER_TESTMODELFACTORY






