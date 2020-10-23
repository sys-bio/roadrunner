
// Module Name
%module(docstring="The RoadRunner SBML Simulation Engine, (c) 2009-2015 J Kyle Medley, Andy Somogyi and Herbert Sauro",
        "threads"=1 /*, directors="1"*/) roadrunner

// ************************************************************
// Module Includes
// ************************************************************

// These are copied directly to the .cxx file and are not parsed
// by SWIG.  Include include files or definitions that are required
// for the module to build correctly.



%{
    #define SWIG_FILE_WITH_INIT
    #include <rrCompiler.h>
    #include <ExecutableModelFactory.h>
    #include <rrExecutableModel.h>
    #include <rrRoadRunnerOptions.h>
    #include <rrRoadRunner.h>
    #include <rrLogger.h>
    #include <rrConfig.h>
    #include <conservation/ConservationExtension.h>
    #include "conservation/ConservedMoietyConverter.h"
    #include "SBMLValidator.h"
    #include "rrSBMLReader.h"
    #include <cstddef>
    #include <map>
    #include <rrVersionInfo.h>
    #include <rrException.h>
    #include <assert.h>
    #include <math.h>
    #include <cmath>


    #include "tr1proxy/cxx11_ns.h"

    using ls::Matrix;
    using ls::DoubleMatrix;
    using ls::Complex;
    using ls::ComplexMatrix;


// Windows is just so special...
#ifdef _WIN32
    #define INFINITY (DBL_MAX + DBL_MAX)
    #define NAN (INFINITY - INFINITY)
    #define isnan _isnan
#else
    #include <signal.h>
    #define isnan std::isnan
#endif

    using namespace rr;

%}




%naturalvar;

// C++ std::string handling
%include "std_string.i"

// C++ std::map handling
%include "std_map.i"

// C++ std::vector handling
%include "std_vector.i"

// not for java
// %include "std_list.i"

//enables better handling of STL exceptions
%include "exception.i"

// correct mapping of unsigned integers
%include "rr_stdint.i"


// the cmake _CMakeLists.txt file in this directory sets the value of the
// SWIG_SHARED_PTR_SUBNAMESPACE as a pre-processor symbol based on the
// USE_TR1_CXX_NS CMake option. SWIG has no way of getting this info
// from the compiler so have to reley on the CMake system.
%include "std_shared_ptr.i"

%shared_ptr(rr::PyIntegratorListener)



%template(IntVector) std::vector<int>;
%template(StringVector) std::vector<std::string>;


%exception {
  try {
    $action
  } catch (const std::exception& e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  }
}


/**
 *  Convert from C --> Python
 *  copy data
 */
%typemap(out) ls::DoubleMatrix {
    // %typemap(out) ls::DoubleMatrix
    const ls::DoubleMatrix* mat = &($1);
    // FIXME
//     $result = doublematrix_to_py(mat, SimulateOptions::COPY_RESULT);
}


/**
 * Convert from C --> Python
 * reference roadrunner owned data.
 */
%typemap(out) const ls::DoubleMatrix* {
    // %typemap(out) const ls::DoubleMatrix*
    const ls::DoubleMatrix* mat = ($1);
//     $result = doublematrix_to_py(mat, 0);
    // FIXME
}

%apply const ls::DoubleMatrix* {ls::DoubleMatrix*, DoubleMatrix*, const DoubleMatrix* };



/* Convert from C --> Python */
%typemap(out) std::vector<double> {

    int len = $1.size();

    std::vector<double>& vec = $1;

//     memcpy(data, &vec[0], sizeof(double)*len);

    // FIXME
}


/* Convert from C --> Python */
%typemap(out) std::vector<ls::Complex> {

    typedef std::complex<double> cpx;

    std::vector<cpx>& vec = $1;

    // FIXME
}



%typemap(out) const rr::Variant& {
    try {
      // FIXME
        const rr::Variant& temp = *($1);
//         $result = Variant_to(temp);
    } catch (const std::exception& e) {
        SWIG_exception(SWIG_RuntimeError, e.what());
    }
}


%typemap(out) const rr::Variant {
    try {
      // FIXME
//         $result = Variant_to($1);
    } catch (const std::exception& e) {
        SWIG_exception(SWIG_RuntimeError, e.what());
    }
}

%apply const rr::Variant {Variant, rr::Variant, const Variant};


%typemap(in) const rr::Variant& (rr::Variant temp) {

    try {
//         temp = Variant_from_?Variant_to_py($input);
        // FIXME
        $1 = &temp;
    } catch (const std::exception& e) {
        SWIG_exception(SWIG_RuntimeError, e.what());
    }
}

%apply const rr::Variant& {rr::Variant&, Variant&, const Variant&};

%{


#ifndef _WIN32

    static void rr_sighandler(int sig) {
        std::cout << "handling signal " << sig << std::endl;
        Log(rr::Logger::LOG_WARNING) << "signal handler : " << sig;
    }

    static unsigned long sigtrap() {
        signal(SIGTRAP, rr_sighandler);
        return raise(SIGTRAP);
    }

#else

    static unsigned long sigtrap() {
        Log(rr::Logger::LOG_WARNING) << "sigtrap not supported on Windows";
        return 0;
    }

#endif
%}





size_t sigtrap();



%{

typedef int (rr::ExecutableModel::*getValuesPtr)(int, int const*, double*);
typedef string (ExecutableModel::*getNamePtr)(int);
typedef int (ExecutableModel::*getNumPtr)();


static std::string strvec_to_pystring(const std::vector<std::string>& strvec) {
    std::stringstream s;
    s << "[";

    for (int i = 0; i < strvec.size(); ++i) {
        s << "'" << strvec[i] << "'";
        if (i + 1 < strvec.size()) {
            s << ",";
        }
    }

    s << "]";

    return s.str();
}

%}

#define LIB_EXTERN
#define RR_DECLSPEC
#define PUGIXML_CLASS

%pragma(java) jniclasscode=%{
  static {
  NativeLib.nativeInit();
  }
%}

%warnfilter(509) rr::RoadRunner::setSelections;
%warnfilter(509) rr::RoadRunner::setSteadyStateSelections;

// Many of the RoadRunner methods will be ignored for the time being
// as currently we do not have a clean mapping to Python.


%ignore rr::RoadRunner::RoadRunner(const std::string&, const std::string&, const std::string&);

%ignore rr::RoadRunner::addCapabilities;
%ignore rr::RoadRunner::getFloatingSpeciesIds;
%ignore rr::RoadRunner::getRateOfChangeIds;
//%ignore rr::RoadRunner::getuCC;
%ignore rr::RoadRunner::addCapability;
%ignore rr::RoadRunner::getFloatingSpeciesInitialConcentrationByIndex;
%ignore rr::RoadRunner::getRatesOfChange;
//%ignore rr::RoadRunner::getuEE;
%ignore rr::RoadRunner::changeInitialConditions;
%ignore rr::RoadRunner::getFloatingSpeciesInitialConcentrations;
%ignore rr::RoadRunner::getRatesOfChangeEx;
%ignore rr::RoadRunner::initializeModel;
%ignore rr::RoadRunner::computeAndAssignConservationLaws;
%ignore rr::RoadRunner::getFloatingSpeciesInitialConditionIds;
%ignore rr::RoadRunner::getReactionIds;
//%ignore rr::RoadRunner::isModelLoaded;
//%ignore rr::RoadRunner::computeSteadyStateValue;
//%ignore rr::RoadRunner::getFullJacobian;
%ignore rr::RoadRunner::getReactionRate;
%ignore rr::RoadRunner::computeSteadyStateValues;
%ignore rr::RoadRunner::getFullReorderedJacobian;
%ignore rr::RoadRunner::getReactionRates;
// %ignore rr::RoadRunner::getOptions;
// %immutable rr::RoadRunner::getOptions;
// Since getOptions returns a reference, SWIG generates a setter as well
%ignore rr::RoadRunner::setOptions; // not needed since getOptions returns a ref
//%ignore rr::RoadRunner::loadSBMLFromFile;
%ignore rr::RoadRunner::correctMaxStep;
//%ignore rr::RoadRunner::getFullyReorderedStoichiometryMatrix;
%ignore rr::RoadRunner::getReactionRatesEx;
%ignore rr::RoadRunner::loadSimulationSettings;
//%ignore rr::RoadRunner::createDefaultSelectionLists;
%ignore rr::RoadRunner::getGlobalParameterByIndex;
//%ignore rr::RoadRunner::getReducedJacobian;
//%ignore rr::RoadRunner::oneStep;
//%ignore rr::RoadRunner::createTimeCourseSelectionList;
%ignore rr::RoadRunner::getGlobalParameterIds;
//%ignore rr::RoadRunner::getReorderedStoichiometryMatrix;
//%ignore rr::RoadRunner::reset;
%ignore rr::RoadRunner::evalModel;
%ignore rr::RoadRunner::getGlobalParameterValues;
//%ignore rr::RoadRunner::getSBML;
%ignore rr::RoadRunner::setBoundarySpeciesByIndex;
%ignore rr::RoadRunner::getBoundarySpeciesAmountIds;
//%ignore rr::RoadRunner::getInfo;
//%ignore rr::RoadRunner::getScaledConcentrationControlCoefficientMatrix;
%ignore rr::RoadRunner::setBoundarySpeciesConcentrations;
%ignore rr::RoadRunner::getBoundarySpeciesByIndex;
//%ignore rr::RoadRunner::getInstanceCount;
//%ignore rr::RoadRunner::getScaledFloatingSpeciesElasticity;
%ignore rr::RoadRunner::setCapabilities;
%ignore rr::RoadRunner::getBoundarySpeciesConcentrations;
//%ignore rr::RoadRunner::getInstanceID;
//%ignore rr::RoadRunner::getScaledFluxControlCoefficientMatrix;
%ignore rr::RoadRunner::setCompartmentByIndex;
%ignore rr::RoadRunner::getBoundarySpeciesIds;
//%ignore rr::RoadRunner::getIntegrator;
//%ignore rr::RoadRunner::getScaledReorderedElasticityMatrix;
%ignore rr::RoadRunner::setCompiler;
//%ignore rr::RoadRunner::getCC;
//%ignore rr::RoadRunner::getL0Matrix;
//%ignore rr::RoadRunner::getSelectedValues;
%ignore rr::RoadRunner::setFloatingSpeciesByIndex;
//%ignore rr::RoadRunner::getCapabilitiesAsXML;
//%ignore rr::RoadRunner::getLinkMatrix;
//%ignore rr::RoadRunner::getSelectionList;
%ignore rr::RoadRunner::setFloatingSpeciesConcentrations;
%ignore rr::RoadRunner::getCapability;
%ignore rr::RoadRunner::getListOfCapabilities;
//%ignore rr::RoadRunner::getSimulationResult;
%ignore rr::RoadRunner::setFloatingSpeciesInitialConcentrationByIndex;
%ignore rr::RoadRunner::getCompartmentByIndex;
%ignore rr::RoadRunner::getListOfParameters;
//%ignore rr::RoadRunner::getSteadyStateSelection;
%ignore rr::RoadRunner::setFloatingSpeciesInitialConcentrations;
%ignore rr::RoadRunner::getCompartmentIds;
//%ignore rr::RoadRunner::getModel;
//%ignore rr::RoadRunner::getSteadyStateSelectionList;
%ignore rr::RoadRunner::setGlobalParameterByIndex;
//%ignore rr::RoadRunner::getCompiler;
//%ignore rr::RoadRunner::getModelGenerator;
//%ignore rr::RoadRunner::getStoichiometryMatrix;
%ignore rr::RoadRunner::setNumPoints;
//%ignore rr::RoadRunner::getConservationMatrix;
%ignore rr::RoadRunner::getModelName;
%ignore rr::RoadRunner::getTempFolder;
%ignore rr::RoadRunner::setParameterValue;
%ignore rr::RoadRunner::getConservedMoietyIds;
//%ignore rr::RoadRunner::getNrMatrix;
//%ignore rr::RoadRunner::getTimeCourseSelectionList;
%ignore rr::RoadRunner::setSimulationSettings;
%ignore rr::RoadRunner::getConservedMoietyValues;
%ignore rr::RoadRunner::getNumPoints;
%ignore rr::RoadRunner::getTimeEnd;
//%ignore rr::RoadRunner::setSteadyStateSelectionList;
//%ignore rr::RoadRunner::getCopyright;
%ignore rr::RoadRunner::getNumberOfBoundarySpecies;
%ignore rr::RoadRunner::getTimeStart;
%ignore rr::RoadRunner::setTempFileFolder;
//%ignore rr::RoadRunner::getDescription;
%ignore rr::RoadRunner::getNumberOfCompartments;
//%ignore rr::RoadRunner::getURL;
//%ignore rr::RoadRunner::setTimeCourseSelectionList;
//%ignore rr::RoadRunner::getEE;
%ignore rr::RoadRunner::getNumberOfDependentSpecies;
//%ignore rr::RoadRunner::getUnscaledConcentrationControlCoefficientMatrix;
%ignore rr::RoadRunner::setTimeEnd;

%ignore rr::RoadRunner::getNumberOfFloatingSpecies;
//%ignore rr::RoadRunner::getUnscaledElasticityMatrix;
%ignore rr::RoadRunner::setTimeStart;

%ignore rr::RoadRunner::getNumberOfGlobalParameters;
//%ignore rr::RoadRunner::getUnscaledFluxControlCoefficientMatrix;
//%ignore rr::RoadRunner::setValue;
%ignore rr::RoadRunner::getNumberOfIndependentSpecies;
//%ignore rr::RoadRunner::getUnscaledSpeciesElasticity;
//%ignore rr::RoadRunner::getExtendedVersionInfo;
%ignore rr::RoadRunner::getNumberOfReactions;
//%ignore rr::RoadRunner::getValue;
//%ignore rr::RoadRunner::steadyState;
%ignore rr::RoadRunner::getValue(const SelectionRecord&);
//%ignore rr::RoadRunner::this;
%ignore rr::RoadRunner::getFloatingSpeciesByIndex;
%ignore rr::RoadRunner::getParameterValue;
//%ignore rr::RoadRunner::getVersion;
%ignore rr::RoadRunner::unLoadModel;
%ignore rr::RoadRunner::getFloatingSpeciesConcentrations;
%ignore rr::RoadRunner::getRateOfChange;
//%ignore rr::RoadRunner::getlibSBMLVersion;
//%ignore rr::RoadRunner::writeSBML;
%ignore rr::RoadRunner::getSimulateOptions;
%ignore rr::RoadRunner::setSimulateOptions;
%ignore rr::RoadRunner::getIds(int types, std::list<std::string> &);

%ignore rr::RoadRunner::getOptions;

%ignore rr::RoadRunner::simulate;

%rename (_load) rr::RoadRunner::load;


%ignore rr::Config::getInt;
%ignore rr::Config::getString;
%ignore rr::Config::getBool;
%ignore rr::Config::getDouble;

%ignore *::setItem;
%ignore *::getItem;
%ignore *::hasKey;
%ignore *::deleteItem;
%ignore *::getKeys;

// ignore SimulateOptions key access methods,
// these are replaced by python dictionary protocol.
//%rename (__setitem__) rr::Dictionary::setItem;
//%rename (__getitem__) rr::Dictionary::getItem;
//%rename (__contains__) rr::Dictionary::hasKey;
//%rename (__delitem__) rr::Dictionary::deleteItem;
//%rename (keys) rr::Dictionary::getKeys;

// do not create these, pure interface.
%nodefaultctor rr::Dictionary;
%nodefaultdtor Dictionary;
%nodefaultctor rr::BasicDictionary;
%nodefaultdtor DictionaryImpl;

// ignore SimulateOptions key access methods,
// these are replaced by python dictionary protocol.
%ignore rr::Integrator::setValue;
%ignore rr::Integrator::getValue;
%ignore rr::Integrator::hasKey;
%ignore rr::Integrator::deleteValue;
%ignore rr::Integrator::getKeys;
%ignore rr::Integrator::setSimulateOptions;
%rename (__str__) rr::Integrator::toString;
%rename (__repr__) rr::Integrator::toRepr;


// rename these, the injected python code will take care of
// making these properties.
%ignore rr::RoadRunner::getSelections();
%ignore rr::RoadRunner::setSelections(const std::vector<rr::SelectionRecord>&);
%rename (_setSelections) setSelections(const std::vector<std::string>&);
%rename (_getModel) getModel();

// hide SelectionRecord details from python api,
// only deal with strings here.
%ignore rr::RoadRunner::getSteadyStateSelections();
%ignore rr::RoadRunner::setSteadyStateSelections(const std::vector<rr::SelectionRecord>&);

%rename (_setSteadyStateSelections) setSteadyStateSelections(const std::vector<std::string>&);
%rename (_getConservedMoietyAnalysis) getConservedMoietyAnalysis();
%rename (_setConservedMoietyAnalysis) setConservedMoietyAnalysis(bool);

%ignore rr::LoggingBuffer;
%ignore rr::LogLevel;
%ignore rr::getLogger;
%ignore rr::ToUpperCase;
%ignore rr::GetHighestLogLevel;
%ignore rr::GetLogLevel;
%ignore rr::GetLogLevelAsString;
%ignore rr::getLogLevel;
%ignore rr::lShowAlways;
%ignore rr::lError;
%ignore rr::lWarning;
%ignore rr::lInfo;
%ignore rr::lDebug;
%ignore rr::lDebug1;
%ignore rr::lDebug2;
%ignore rr::lDebug3;
%ignore rr::lDebug4;
%ignore rr::lDebug5;
%ignore rr::lAny;
%ignore rr::lUser;


%ignore rr::ExecutableModel::getFloatingSpeciesAmounts(int, int const*, double *);
%ignore rr::ExecutableModel::setFloatingSpeciesAmounts(int len, int const *indx, const double *values);
%ignore rr::ExecutableModel::getFloatingSpeciesAmountRates(int, int const*, double *);
%ignore rr::ExecutableModel::getFloatingSpeciesConcentrationRates(int, int const*, double *);

%ignore rr::ExecutableModel::getFloatingSpeciesConcentrations(int, int const*, double *);
%ignore rr::ExecutableModel::setFloatingSpeciesConcentrations(int len, int const *indx, const double *values);
%ignore rr::ExecutableModel::setFloatingSpeciesInitConcentrations(int len, int const *indx, const double *values);
%ignore rr::ExecutableModel::getFloatingSpeciesInitConcentrations(int, int const*, double *);
%ignore rr::ExecutableModel::getBoundarySpeciesAmounts(int, int const*, double *);
%ignore rr::ExecutableModel::getBoundarySpeciesConcentrations(int, int const*, double *);
%ignore rr::ExecutableModel::setBoundarySpeciesConcentrations(int len, int const *indx, const double *values);
%ignore rr::ExecutableModel::getGlobalParameterValues(int, int const*, double *);
%ignore rr::ExecutableModel::setGlobalParameterValues(int len, int const *indx, const double *values);
%ignore rr::ExecutableModel::getCompartmentVolumes(int, int const*, double *);
%ignore rr::ExecutableModel::setCompartmentVolumes(int len, int const *indx, const double *values);
%ignore rr::ExecutableModel::getConservedMoietyValues(int, int const*, double *);
%ignore rr::ExecutableModel::setConservedMoietyValues(int len, int const *indx, const double *values);
%ignore rr::ExecutableModel::getReactionRates(int, int const*, double *);
%ignore rr::ExecutableModel::evalReactionRates;
%ignore rr::ExecutableModel::convertToAmounts;
%ignore rr::ExecutableModel::computeConservedTotals;
%ignore rr::ExecutableModel::setRateRuleValues;
%ignore rr::ExecutableModel::getRateRuleValues;
%ignore rr::ExecutableModel::getStateVector(double *stateVector);
%ignore rr::ExecutableModel::setStateVector;
%ignore rr::ExecutableModel::convertToConcentrations;
%ignore rr::ExecutableModel::updateDependentSpeciesValues;
%ignore rr::ExecutableModel::computeAllRatesOfChange;
%ignore rr::ExecutableModel::getStateVectorRate(double time, const double *y, double* dydt);
%ignore rr::ExecutableModel::getStateVectorRate(double time, const double *y);
%ignore rr::ExecutableModel::testConstraints;
%ignore rr::ExecutableModel::print;
//%ignore rr::ExecutableModel::getNumEvents;
%ignore rr::ExecutableModel::getEventTriggers;
%ignore rr::ExecutableModel::evalEvents;
%ignore rr::ExecutableModel::applyPendingEvents;
%ignore rr::ExecutableModel::evalEventRoots;
%ignore rr::ExecutableModel::getNextPendingEventTime;
%ignore rr::ExecutableModel::getPendingEventSize;
%ignore rr::ExecutableModel::resetEvents;

%ignore rr::ExecutableModel::getFloatingSpeciesId(int index);
%ignore rr::ExecutableModel::getBoundarySpeciesId(int index);
%ignore rr::ExecutableModel::getGlobalParameterId(int index);
%ignore rr::ExecutableModel::getCompartmentId(int index);
%ignore rr::ExecutableModel::getConservedMoietyId(int index);
%ignore rr::ExecutableModel::getReactionId(int index);

%ignore rr::ExecutableModel::getFloatingSpeciesIndex(const std::string& eid);
%ignore rr::ExecutableModel::getBoundarySpeciesIndex(const std::string &eid);
%ignore rr::ExecutableModel::getGlobalParameterIndex(const std::string& eid);
%ignore rr::ExecutableModel::getCompartmentIndex(const std::string& eid);
%ignore rr::ExecutableModel::getConservedMoietyIndex(const std::string& eid);
%ignore rr::ExecutableModel::getReactionIndex(const std::string& eid);

%ignore rr::ExecutableModel::getStoichiometryMatrix(int*, int*, double**);

%ignore rr::ExecutableModel::getConservedSumChanged();
%ignore rr::ExecutableModel::computeConservedTotals();
%ignore rr::ExecutableModel::setConservedSumChanged(bool);
%ignore rr::ExecutableModel::convertToAmounts() ;

%ignore rr::ExecutableModel::setFloatingSpeciesInitConcentrations(int len, int const *indx, double const *values);
%ignore rr::ExecutableModel::getFloatingSpeciesInitConcentrations(int len, int const *indx, double *values);
%ignore rr::ExecutableModel::setFloatingSpeciesInitAmounts(int len, int const *indx, double const *values);
%ignore rr::ExecutableModel::getFloatingSpeciesInitAmounts(int len, int const *indx, double *values);
%ignore rr::ExecutableModel::setCompartmentInitVolumes(int len, int const *indx, double const *values);
%ignore rr::ExecutableModel::getCompartmentInitVolumes(int len, int const *indx, double *values);
%ignore rr::ExecutableModel::getIds(int, std::list<std::string> &);

// deprecated, model knows how to reset itself with reset.
%ignore rr::ExecutableModel::evalInitialConditions;

// map the events to python using the PyEventListener class
%ignore rr::ExecutableModel::setEventListener(int, rr::EventListenerPtr);
%ignore rr::ExecutableModel::getEventListener(int);
%ignore rr::EventListenerPtr;
%ignore rr::EventListenerException;

// ignore the EventListener virtuals, but leave the enum
%ignore rr::EventListener::onTrigger(ExecutableModel* model, int eventIndex, const std::string& eventId);
%ignore rr::EventListener::onAssignment(ExecutableModel* model, int eventIndex, const std::string& eventId);

// ignore the C++ class, only deal with the python version
%ignore rr::IntegratorListener;

%ignore rr::Integrator::setListener(rr::IntegratorListenerPtr);
%ignore rr::Integrator::getListener();

//%ignore rr::Integrator::addIntegratorListener;
//%ignore rr::Integrator::removeIntegratorListener;

// ignore integrator registration

%ignore rr::IntegratorRegistrar;
%ignore rr::IntegratorFactory;

%rename rr::conversion::ConservedMoietyConverter JConservedMoietyConverter;

%ignore rr::ostream;
%ignore ostream;
%ignore std::ostream;
%ignore operator<<(ostream&, const rr::SelectionRecord& rec);
%ignore operator<<(rr::ostream&, const rr::SelectionRecord& rec);

// RoadRunner class proxies

//%typemap(javaimports) rr::RoadRunner %{
//  import org.la4j.Matrices;
//  import org.la4j.matrix.MatrixFactory;
//  import org.la4j.Matrix;
//  import org.la4j.matrix.DenseMatrix;
//%}

%typemap(javacode) rr::RoadRunner %{
  public LabeledData simulate(double tstart, double tend, int n) {
    double[][] simresult = (double[][])roadrunner.jrr_simulate_(this, tstart, tend, n);
    LabeledData d = new LabeledData(simresult);
    return d;
  }
%}


//%ignore rr::DictionaryImpl;

// Warning 389: operator[] ignored (consider using %extend)
// Warning 401: Nothing known about base class 'Configurable'. Ignored.


// Warning 315: Nothing known about 'std::ostream'.
namespace std { class ostream{}; }


// Warning 401: Nothing known about base class 'Configurable'. Ignored.


/**
 * include the roadrunner files here, this is where the wrappers are generated.
 */


/**
 * this returns a new object
 */
%newobject rr::ExecutableModelFactory::createModel;



%include <Dictionary.h>
%include <rrRoadRunnerOptions.h>
%include <rrLogger.h>
%include <rrCompiler.h>
%include <rrExecutableModel.h>
%include <ExecutableModelFactory.h>
%include <rrVersionInfo.h>

%include <rrRoadRunner.h>


%include <rrSelectionRecord.h>
%include <conservation/ConservedMoietyConverter.h>
%include <Integrator.h>

// %include "PyEventListener.h"
// %include "PyIntegratorListener.h"
%include <rrConfig.h>
%include <SBMLValidator.h>
%include <rrSBMLReader.h>

// TODO: instead use pragmas to insert code into proxy classes (see http://swig-user.narkive.com/aiRVVhtk/including-custom-java-code)
// http://stackoverflow.com/questions/10112934/return-java-lang-object-using-swig-native-function
// http://stackoverflow.com/questions/6143134/return-a-2d-primitive-array-from-c-to-java-from-jni-ndk
%native(jrr_simulate_) jobjectArray jrr_simulate_(rr::RoadRunner* rr, double tstart, double tend, int n);
//%inline %{
//  int jrr_simulate_(rr::RoadRunner* rr, double tstart, double tend, int n);
//%}

%{

#ifdef __cplusplus
extern "C" {
#endif

SWIGEXPORT jobjectArray JNICALL Java_roadrunner_roadrunnerJNI_jrr_1simulate_1(JNIEnv *jenv, jclass jcls, jlong jr, jobject jr_, jdouble jtstart, jdouble jtend, jint jn) {
  jobjectArray jresult = 0 ;
  rr::RoadRunner *r = (rr::RoadRunner *) 0 ;
  double tstart ;
  double tend ;
  int n;
  const DoubleMatrix* mat;

  int rows, cols;

  (void)jenv;
  (void)jcls;
  (void)jr_;
  r = *(rr::RoadRunner **)&jr;
  tstart = (double)jtstart;
  tend = (double)jtend;
  n = (int)jn;

  rr::SimulateOptions& opt = r->getSimulateOptions();
  opt.start = tstart;
  opt.duration = tend - tstart;
  opt.steps = n;
  //if(variable step?) {
  //  opt.integratorFlags |= Integrator::VARIABLE_STEP;
  //}

  // call RoadRunner method
  {
    try {
      mat = r->simulate();
    } catch (const std::exception& e) {
      {
        SWIG_JavaException(jenv, SWIG_RuntimeError, e.what()); return 0;
      };
    }
  }

  // copy output to Java array

  rows = mat->numRows();
  cols = mat->numCols();

  jclass doubleArrayClass = jenv->FindClass("[D");
  if(!doubleArrayClass) {
    return NULL;
  }

  jresult = (jobjectArray) jenv->NewObjectArray(rows, doubleArrayClass, NULL);
  for(int k=0; k<rows; ++k) {
    // LS Matrix operator[] gets row
    jdoubleArray subarray = jenv->NewDoubleArray(cols);
    jenv->SetDoubleArrayRegion(subarray, (jsize)0, (jsize) cols, (jdouble*)(*mat)[k]);
    jenv->SetObjectArrayElement(jresult, (jsize)k, subarray);
    jenv->DeleteLocalRef(subarray);
  }

  return jresult;
}

#ifdef __cplusplus
}
#endif

%}

%extend rr::RoadRunner
{
  // attributes

  /**
    * make some of these const so SWIG would not allow setting.
    */
  const rr::SimulateOptions *simulateOptions;

  rr::RoadRunnerOptions *options;

  std::string __repr__() {
      std::stringstream s;
      s << "<roadrunner.RoadRunner() { this = " << (void*)$self << " }>";
      return s.str();
  }

  std::string __str__() {
      return $self->getInfo();
  }

  double getValue(const rr::SelectionRecord* pRecord) {
      return $self->getValue(*pRecord);
  }

  double __getitem__(const std::string& id) {
      return ($self)->getValue(id);
  }

  void __setitem__(const std::string& id, double value) {
      ($self)->setValue(id, value);
  }

  int simulate_() {
    return 0;
  }
}

%{
    rr::SimulateOptions* rr_RoadRunner_simulateOptions_get(RoadRunner* r) {
        return &r->getSimulateOptions();
    }

    void rr_RoadRunner_simulateOptions_set(RoadRunner* r, const rr::SimulateOptions* opt) {
        r->setSimulateOptions(*opt);
    }


    rr::RoadRunnerOptions* rr_RoadRunner_options_get(RoadRunner* r) {
        return &r->getOptions();
    }

    void rr_RoadRunner_options_set(RoadRunner* r, const rr::RoadRunnerOptions* opt) {
        rr::RoadRunnerOptions *rropt = &r->getOptions();
        *rropt = *opt;
    }
%}


%extend rr::SelectionRecord {

    std::string __repr__() {
        return $self->to_repr();
    }

    std::string __str__() {
        return $self->to_string();
    }
}

%extend rr::SimulateOptions
{
    double end;
    bool resetModel;
    bool stiff;
    bool multiStep;
    bool structuredResult;
    bool variableStep;
    bool copyResult;

    std::string __repr__() {
        return ($self)->toRepr();
    }


    std::string __str__() {
        return ($self)->toString();
    }
}

%{
    double rr_SimulateOptions_end_get(SimulateOptions* opt) {
        return opt->start + opt->duration;
    }

    void rr_SimulateOptions_end_set(SimulateOptions* opt, double end) {
        opt->duration = end - opt->start;
    }

    bool rr_SimulateOptions_resetModel_get(SimulateOptions* opt) {
        return opt->getItem("reset");
    }

    void rr_SimulateOptions_resetModel_set(SimulateOptions* opt, bool value) {
        opt->setItem("reset", value);
    }

    bool rr_SimulateOptions_structuredResult_get(SimulateOptions* opt) {
        return opt->structured_result;
    }

    void rr_SimulateOptions_structuredResult_set(SimulateOptions* opt, bool value) {
        opt->structured_result = value;
    }

    bool rr_SimulateOptions_stiff_get(SimulateOptions* opt) {
        return opt->getItem("stiff");
    }

    void rr_SimulateOptions_stiff_set(SimulateOptions* opt, bool value) {
        opt->setItem("stiff", value);
    }

    bool rr_SimulateOptions_multiStep_get(SimulateOptions* opt) {
        return opt->getItem("multiStep");
    }

    void rr_SimulateOptions_multiStep_set(SimulateOptions* opt, bool value) {
        opt->setItem("multiStep", value);
    }

    bool rr_SimulateOptions_copyResult_get(SimulateOptions* opt) {
        return opt->getItem("copyResult");
    }

    void rr_SimulateOptions_copyResult_set(SimulateOptions* opt, bool value) {
        opt->setItem("copyResult", value);
    }

    bool rr_SimulateOptions_variableStep_get(SimulateOptions* opt) {
        return opt->getItem("variableStep");
    }

    void rr_SimulateOptions_variableStep_set(SimulateOptions* opt, bool value) {
        opt->setItem("variableStep", value);
    }
%}



%extend rr::RoadRunnerOptions
{
    bool disablePythonDynamicProperties;
}

%{


    bool rr_RoadRunnerOptions_disablePythonDynamicProperties_get(RoadRunnerOptions* opt) {
        return opt->flags & rr::RoadRunnerOptions::DISABLE_PYTHON_DYNAMIC_PROPERTIES;
    }

    void rr_RoadRunnerOptions_disablePythonDynamicProperties_set(RoadRunnerOptions* opt, bool value) {
        if (value) {
            opt->flags |= rr::RoadRunnerOptions::DISABLE_PYTHON_DYNAMIC_PROPERTIES;
        } else {
            opt->flags &= ~rr::RoadRunnerOptions::DISABLE_PYTHON_DYNAMIC_PROPERTIES;
        }
    }


%}

%extend rr::LoadSBMLOptions
{
    bool conservedMoieties;
    bool mutableInitialConditions;
    bool noDefaultSelections;
    bool readOnly;
    bool recompile;
}


%{
    bool rr_LoadSBMLOptions_conservedMoieties_get(rr::LoadSBMLOptions* opt) {
        return opt->modelGeneratorOpt & rr::LoadSBMLOptions::CONSERVED_MOIETIES;
    }


    void rr_LoadSBMLOptions_conservedMoieties_set(rr::LoadSBMLOptions* opt, bool value) {
        if (value) {
            opt->modelGeneratorOpt |= rr::LoadSBMLOptions::CONSERVED_MOIETIES;
        } else {
            opt->modelGeneratorOpt &= ~rr::LoadSBMLOptions::CONSERVED_MOIETIES;
        }
    }


    bool rr_LoadSBMLOptions_noDefaultSelections_get(rr::LoadSBMLOptions* opt) {
        return opt->loadFlags & rr::LoadSBMLOptions::NO_DEFAULT_SELECTIONS;
    }

    void rr_LoadSBMLOptions_noDefaultSelections_set(rr::LoadSBMLOptions* opt, bool value) {
        if (value) {
            opt->loadFlags |= rr::LoadSBMLOptions::NO_DEFAULT_SELECTIONS;
        } else {
            opt->loadFlags &= ~rr::LoadSBMLOptions::NO_DEFAULT_SELECTIONS;
        }
    }

    bool rr_LoadSBMLOptions_mutableInitialConditions_get(rr::LoadSBMLOptions* opt) {
        return opt->modelGeneratorOpt & rr::LoadSBMLOptions::MUTABLE_INITIAL_CONDITIONS;
    }


    void rr_LoadSBMLOptions_mutableInitialConditions_set(rr::LoadSBMLOptions* opt, bool value) {
        if (value) {
            opt->modelGeneratorOpt |= rr::LoadSBMLOptions::MUTABLE_INITIAL_CONDITIONS;
        } else {
            opt->modelGeneratorOpt &= ~rr::LoadSBMLOptions::MUTABLE_INITIAL_CONDITIONS;
        }
    }

    bool rr_LoadSBMLOptions_recompile_get(rr::LoadSBMLOptions* opt) {
        return opt->modelGeneratorOpt & rr::LoadSBMLOptions::RECOMPILE;
    }


    void rr_LoadSBMLOptions_recompile_set(rr::LoadSBMLOptions* opt, bool value) {
        if (value) {
            opt->modelGeneratorOpt |= rr::LoadSBMLOptions::RECOMPILE;
        } else {
            opt->modelGeneratorOpt &= ~rr::LoadSBMLOptions::RECOMPILE;
        }
    }

    bool rr_LoadSBMLOptions_readOnly_get(rr::LoadSBMLOptions* opt) {
        return opt->modelGeneratorOpt & rr::LoadSBMLOptions::READ_ONLY;
    }


    void rr_LoadSBMLOptions_readOnly_set(rr::LoadSBMLOptions* opt, bool value) {
        if (value) {
            opt->modelGeneratorOpt |= rr::LoadSBMLOptions::READ_ONLY;
        } else {
            opt->modelGeneratorOpt &= ~rr::LoadSBMLOptions::READ_ONLY;
        }

    }
%}




%extend rr::ExecutableModel
{
    /***
     ** set values section
     ***/

    int setFloatingSpeciesAmounts(int len, double const *values) {
        return $self->setFloatingSpeciesAmounts(len, 0, values);
    }

    int setFloatingSpeciesConcentrations(int len, double const *values) {
        return $self->setFloatingSpeciesConcentrations(len, 0, values);
    }

    int setBoundarySpeciesConcentrations(int len, double const *values) {
        return $self->setBoundarySpeciesConcentrations(len, 0, values);
    }

    int setGlobalParameterValues(int len, double const *values) {
        return $self->setGlobalParameterValues(len, 0, values);
    }

    int setCompartmentVolumes(int len, double const *values) {
        return $self->setCompartmentVolumes(len, 0, values);
    }

    int setConservedMoietyValues(int len, double const *values) {
        return $self->setConservedMoietyValues(len, 0, values);
    }

    int setFloatingSpeciesInitConcentrations(int len, double const *values) {
        return $self->setFloatingSpeciesInitConcentrations(len, 0, values);
    }

    int setFloatingSpeciesInitAmounts(int len, double const *values) {
        return $self->setFloatingSpeciesInitAmounts(len, 0, values);
    }

    int setCompartmentInitVolumes(int len, double const *values) {
        return $self->setCompartmentInitVolumes(len, 0, values);
    }


    int setFloatingSpeciesAmounts(int leni, int const* indx, int lenv, double const *values) {
        if (leni != lenv) {
            // exception
            return -1;
        }
        return $self->setFloatingSpeciesAmounts(leni, indx, values);
    }


    int setFloatingSpeciesConcentrations(int leni, int const* indx, int lenv, double const *values) {
        if (leni != lenv) {
            // exception
            return -1;
        }
        return $self->setFloatingSpeciesConcentrations(leni, indx, values);
    }

    int setBoundarySpeciesConcentrations(int leni, int const* indx, int lenv, double const *values) {
        if (leni != lenv) {
            // exception
            return -1;
        }
        return $self->setBoundarySpeciesConcentrations(leni, indx, values);
    }

    int setGlobalParameterValues(int leni, int const* indx, int lenv, double const *values) {
        if (leni != lenv) {
            // exception
            return -1;
        }
        return $self->setGlobalParameterValues(leni, indx, values);
    }

    int setCompartmentVolumes(int leni, int const* indx, int lenv, double const *values) {
        if (leni != lenv) {
            // exception
            return -1;
        }
        return $self->setCompartmentVolumes(leni, indx, values);
    }

    int setConservedMoietyValues(int leni, int const* indx, int lenv, double const *values) {
        if (leni != lenv) {
            // exception
            return -1;
        }
        return $self->setConservedMoietyValues(leni, indx, values);
    }

    int setFloatingSpeciesInitConcentrations(int leni, int const* indx, int lenv, double const *values) {
        if (leni != lenv) {
            // exception
            return -1;
        }
        return $self->setFloatingSpeciesInitConcentrations(leni, indx, values);
    }


    int setFloatingSpeciesInitAmounts(int leni, int const* indx, int lenv, double const *values) {
        if (leni != lenv) {
            // exception
            return -1;
        }
        return $self->setFloatingSpeciesInitAmounts(leni, indx, values);
    }


    int setCompartmentInitVolumes(int leni, int const* indx, int lenv, double const *values) {
        if (leni != lenv) {
            // exception
            return -1;
        }
        return $self->setCompartmentInitVolumes(leni, indx, values);
    }


    int setFloatingSpeciesInitConcentrations(int leni, int const* indx, int lenv, double const *values) {
        if (leni != lenv) {
            // exception
            return -1;
        }
        return $self->setFloatingSpeciesInitConcentrations(leni, indx, values);
    }

    int setFloatingSpeciesInitAmounts(int leni, int const* indx, int lenv, double const *values) {
        if (leni != lenv) {
            // exception
            return -1;
        }
        return $self->setFloatingSpeciesInitAmounts(leni, indx, values);
    }


    int setCompartmentInitVolumes(int leni, int const* indx, int lenv, double const *values) {
        if (leni != lenv) {
            // exception
            return -1;
        }
        return $self->setCompartmentInitVolumes(leni, indx, values);
    }
}

%extend rr::Integrator {
}





