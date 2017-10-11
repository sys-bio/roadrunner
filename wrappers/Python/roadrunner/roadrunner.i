
// Module Name
%module(docstring="The RoadRunner SBML Simulation Engine, (c) 2009-2014 Andy Somogyi and Herbert Sauro",
        "threads"=1 /*, directors="1"*/) roadrunner

// most methods should leave the GIL locked, no point to extra overhead
// for fast methods. Only Roadrunner with long methods like simulate
// and load release the GIL.
%nothread;

//%feature("director") PyEventListener;

// ************************************************************
// Module Includes
// ************************************************************

// These are copied directly to the .cxx file and are not parsed
// by SWIG.  Include include files or definitions that are required
// for the module to build correctly.



%{
    #define SWIG_FILE_WITH_INIT
    // see discussion on import array,
    // http://docs.scipy.org/doc/numpy/reference/c-api.array.html#miscellaneous
    #define PY_ARRAY_UNIQUE_SYMBOL RoadRunner_ARRAY_API
    #include <numpy/arrayobject.h>
    #include <lsMatrix.h>
    #include <lsLibla.h>
    #include <lsLA.h>
    #include <lsLUResult.h>
    #include <lsUtils.h>
    #include <rrCompiler.h>
    #include <ExecutableModelFactory.h>
    #include <rrExecutableModel.h>
    #include <rrRoadRunnerOptions.h>
    #include <rrRoadRunner.h>
    #include <SteadyStateSolver.h>
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
    #include "PyUtils.h"
    #include "PyLoggerStream.h"

    // make a python obj out of the C++ ExecutableModel, this is used by the PyEventListener
    // class. This function is defined later in this compilation unit.
    PyObject *ExecutableModel_NewPythonObj(rr::ExecutableModel*);
    PyObject *Integrator_NewPythonObj(rr::Integrator*);


    #include "PyEventListener.h"
    #include "PyIntegratorListener.h"

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

#define VERIFY_PYARRAY(p) { \
    assert(p && "PyArray is NULL"); \
    assert((PyArray_NBYTES(p) > 0 ? PyArray_ISCARRAY(p) : true) &&  "PyArray must be C format"); \
}


    class DictionaryHolder {
    public:
        rr::Dictionary* dict;

        DictionaryHolder() { dict = NULL; }

        ~DictionaryHolder() {
            Log(Logger::LOG_TRACE) << __FUNC__ << ", deleting dictionary " << (void*)dict;
            delete dict;
        }
    };


%}




%naturalvar;

// C++ std::string handling
%include "std_string.i"

// C++ std::map handling
%include "std_map.i"

// C++ std::map handling
%include "std_vector.i"

%include "std_list.i"

//enables better handling of STL exceptions
%include "exception.i"

// correct mapping of unsigned integers
%include "rr_stdint.i"

// all the documentation goes here.
%include "rr_docstrings.i"


// the cmake CMakeLists.txt file in this directory sets the value of the
// SWIG_SHARED_PTR_SUBNAMESPACE as a pre-processor symbol based on the
// USE_TR1_CXX_NS CMake option. SWIG has no way of getting this info
// from the compiler so have to reley on the CMake system.
%include "std_shared_ptr.i"

%shared_ptr(rr::PyIntegratorListener)



%template(IntVector) std::vector<int>;
%template(StringVector) std::vector<std::string>;
%template(StringList) std::list<std::string>;

%apply std::vector<std::string> {vector<std::string>, vector<string>, std::vector<string> };

//%template(SelectionRecordVector) std::vector<rr::SelectionRecord>;
//%apply std::vector<rr::SelectionRecord> {std::vector<SelectionRecord>, std::vector<rr::SelectionRecord>, vector<SelectionRecord>};

%apply std::list<std::string>& OUTPUT {std::list<std::string>};

%template(DictionaryVector) std::vector<const rr::Dictionary*>;
%apply std::vector<const rr::Dictionary*> {std::vector<const Dictionary*>, vector<const rr::Dictionary*>, vector<const Dictionary*>};

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
    $result = doublematrix_to_py(mat, false, true);
}


/**
 * Convert from C --> Python
 * reference roadrunner owned data.
 */
%typemap(out) const ls::DoubleMatrix* {
    // %typemap(out) const ls::DoubleMatrix*
    const ls::DoubleMatrix* mat = ($1);
    $result = doublematrix_to_py(mat, false, false);
}

%apply const ls::DoubleMatrix* {ls::DoubleMatrix*, DoubleMatrix*, const DoubleMatrix* };



/* Convert from C --> Python */
%typemap(out) std::vector<double> {

    int len = $1.size();
    npy_intp dims[1] = {len};

    PyObject *array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    VERIFY_PYARRAY(array);

    if (!array) {
        // TODO error handling.
        return 0;
    }

    double *data = (double*)PyArray_DATA((PyArrayObject*)array);

    std::vector<double>& vec = $1;

    memcpy(data, &vec[0], sizeof(double)*len);

    $result  = array;
}


/* Convert from C --> Python */
%typemap(out) std::vector<ls::Complex> {

    typedef std::complex<double> cpx;

    std::vector<cpx>& vec = $1;

    bool iscpx = false;

    // small number
    double epsilon = 2 * std::numeric_limits<double>::epsilon();
    for (std::vector<cpx>::const_iterator i = vec.begin(); i != vec.end(); ++i)
    {
        iscpx = iscpx || (std::imag(*i) >= epsilon);
        if (iscpx) break;
    }

    if (iscpx) {
        int len = $1.size();
        npy_intp dims[1] = {len};

        PyObject *array = PyArray_SimpleNew(1, dims, NPY_COMPLEX128);
        VERIFY_PYARRAY(array);

        if (!array) {
            // TODO error handling.
            return 0;
        }

        cpx *data = (cpx*)PyArray_DATA((PyArrayObject*)array);

        memcpy(data, &vec[0], sizeof(std::complex<double>)*len);

        $result  = array;
    } else {
        int len = $1.size();
        npy_intp dims[1] = {len};

        PyObject *array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        VERIFY_PYARRAY(array);

        if (!array) {
            // TODO error handling.
            return 0;
        }

        double *data = (double*)PyArray_DATA((PyArrayObject*)array);

        for (int i = 0; i < vec.size(); ++i) {
            data[i] = std::real(vec[i]);
        }

        $result  = array;
    }
}



%typemap(out) const rr::Variant& {
    try {
        const rr::Variant& temp = *($1);
        $result = Variant_to_py(temp);
    } catch (const std::exception& e) {
        SWIG_exception(SWIG_RuntimeError, e.what());
    }
}


%typemap(out) const rr::Variant {
    try {
        $result = Variant_to_py($1);
    } catch (const std::exception& e) {
        SWIG_exception(SWIG_RuntimeError, e.what());
    }
}

%apply const rr::Variant {Variant, rr::Variant, const Variant};


%typemap(in) const rr::Variant& (rr::Variant temp) {

    try {
        temp = Variant_from_py($input);
        $1 = &temp;
    } catch (const std::exception& e) {
        SWIG_exception(SWIG_RuntimeError, e.what());
    }
}

%apply const rr::Variant& {rr::Variant&, Variant&, const Variant&};


/**
 * input map, convert an incomming object to a roadrunner Dictionary*
 */
%typemap(in) const rr::Dictionary* (DictionaryHolder holder, void* argp) {

    try {
        // check if null, this is fine,
        if($input == NULL) {
            $1 = NULL;
        }
        else {
            // first check if its a roadrunner type
            int res = SWIG_ConvertPtr($input, &argp,SWIGTYPE_p_rr__Dictionary, 0 |  0 );
            if (SWIG_IsOK(res)) {
                $1 = reinterpret_cast< rr::Dictionary * >(argp);
            } else {
                holder.dict = Dictionary_from_py($input);
                $1 = holder.dict;
            }
        }
    } catch (const std::exception& e) {
        SWIG_exception(SWIG_RuntimeError, e.what());
    }
}

%typemap(typecheck) const rr::Dictionary* = PyObject*;

%apply const rr::Dictionary* {const Dictionary*, rr::Dictionary*, Dictionary*};







/*
%typemap(out) std::vector<std::string> {

    int len = $1.size();

    PyObject* pyList = PyList_New(len);

    for(int i = 0; i < len; i++)
    {
        const std::string& str  = $1.at(i);
        PyObject* pyStr = PyString_FromString(str.c_str());
        PyList_SET_ITEM(pyList, i, pyStr);
    }

    $result = pyList;
}
*/

//%apply std::vector<std::string> {vector<std::string>, vector<string>, std::vector<string> };






%include "numpy.i"


%init %{
import_array();
rr::pyutil_init(m);
%}

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


static PyObject* _ExecutableModel_getValues(rr::ExecutableModel *self, getValuesPtr func,
                                            getNumPtr numPtr, int len, int const *indx) {
    if (len <= 0) {
        len = (self->*numPtr)();
        indx = 0;
    }

    npy_intp dims[1] = {len};
    PyObject *array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    VERIFY_PYARRAY(array);

    if (!array) {
        // TODO error handling.
        return 0;
    }

    double *data = (double*)PyArray_DATA((PyArrayObject*)array);

    (self->*func)(len, indx, data);

    // TODO check result
    return array;
}



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



// make a python obj out of the C++ ExecutableModel, this is used by the PyEventListener
// class. This function is defined later in this compilation unit.

PyObject *ExecutableModel_NewPythonObj(rr::ExecutableModel* e) {
    return SWIG_NewPointerObj(SWIG_as_voidptr(e), SWIGTYPE_p_rr__ExecutableModel, 0 |  0 );
}

PyObject *Integrator_NewPythonObj(rr::Integrator* i) {
    return SWIG_NewPointerObj(SWIG_as_voidptr(i), SWIGTYPE_p_rr__Integrator, 0 |  0 );
}



%}


%apply (int DIM1, int* IN_ARRAY1) {(int len, int const *indx)};

%apply (int DIM1, double* IN_ARRAY1) {(int len, double const *values)};

// typemap for the set***Values methods
%apply (int DIM1, int* IN_ARRAY1) {(int leni, int const* indx)};
%apply (int DIM1, double* IN_ARRAY1) {(int lenv, const  double* values)};

// typemap for getStateVector, getStateVectorRate
%apply (int DIM1, double* IN_ARRAY1)      {(int in_len, double const *in_values)};
%apply (int DIM1, double* INPLACE_ARRAY1) {(int out_len, double* out_values)};

#define LIB_EXTERN
#define RR_DECLSPEC
#define PUGIXML_CLASS



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
//%ignore rr::RoadRunner::getConservedMoietyIds;
//%ignore rr::RoadRunner::getNrMatrix;
//%ignore rr::RoadRunner::getTimeCourseSelectionList;
%ignore rr::RoadRunner::setSimulationSettings;
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
//%ignore rr::RoadRunner::simulate;
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
//%ignore rr::RoadRunner::getFloatingSpeciesConcentrations;
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
//%ignore rr::Integrator::setValue;
//%ignore rr::Integrator::getValue;
%ignore rr::Integrator::hasKey;
%ignore rr::Integrator::deleteValue;
%ignore rr::Integrator::getKeys;
%ignore rr::Integrator::setSimulateOptions;
%rename (__str__) rr::Integrator::toString;
//%rename (__repr__) rr::Integrator::toRepr;


%rename (__str__) rr::SteadyStateSolver::toString;

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
//%ignore rr::RoadRunner::getSteadyStateValues();
//%rename (getSteadyStateValues) rr::RoadRunner::getSteadyStateValuesNamedArray();
//%rename (steadyState) rr::RoadRunner::steadyState;
//%rename (steadyState) rr::RoadRunner::steadyStateNamedArray;

%rename (_setSteadyStateSelections) setSteadyStateSelections(const std::vector<std::string>&);
%rename (_getConservedMoietyAnalysis) rr::RoadRunner::getConservedMoietyAnalysis();
%rename (_setConservedMoietyAnalysis) rr::RoadRunner::setConservedMoietyAnalysis(bool);

// Swig wraps C++ vectors to tuples, need to wrap lists instead on some methods
%rename (_getIndependentFloatingSpeciesIds) rr::RoadRunner::getIndependentFloatingSpeciesIds();
%rename (_getDependentFloatingSpeciesIds) rr::RoadRunner::getDependentFloatingSpeciesIds();

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

%rename rr::conversion::ConservedMoietyConverter PyConservedMoietyConverter;

%ignore rr::ostream;
%ignore ostream;
%ignore std::ostream;
%ignore operator<<(ostream&, const rr::SelectionRecord& rec);
%ignore operator<<(rr::ostream&, const rr::SelectionRecord& rec);



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

%thread;
%include <rrRoadRunner.h>
%nothread;

%include <rrSelectionRecord.h>
%include <conservation/ConservedMoietyConverter.h>
%include <Solver.h>
%include <Integrator.h>
%include <SteadyStateSolver.h>

%include "PyEventListener.h"
%include "PyIntegratorListener.h"
%include <rrConfig.h>
%include <SBMLValidator.h>
%include <rrSBMLReader.h>


%extend rr::RoadRunner
{
    // attributes

    /**
     * make some of these const so SWIG would not allow setting.
     */
    const rr::SimulateOptions *__simulateOptions;

    rr::RoadRunnerOptions *options;

    std::string __repr__() {
        std::stringstream s;
        s << "<roadrunner.RoadRunner() { this = " << (void*)$self << " }>";
        return s.str();
    }

    std::string __str__() {
        return $self->getInfo();
    }

    PyObject* _simulate(const rr::SimulateOptions* opt) {
        // its not const correct...
        ls::DoubleMatrix *result = const_cast<ls::DoubleMatrix*>($self->simulate(opt));

        return doublematrix_to_py(result, opt->structured_result, opt->copy_result);
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

    PyObject *getIds(int types) {
        std::list<std::string> ids;

        ($self)->getIds(types, ids);

        unsigned size = ids.size();

        PyObject* pyList = PyList_New(size);

        unsigned j = 0;

        for (std::list<std::string>::const_iterator i = ids.begin(); i != ids.end(); ++i)
        {
            const std::string& id  = *i;
            PyObject* pyStr = PyString_FromString(id.c_str());
            PyList_SET_ITEM(pyList, j++, pyStr);
        }

        return pyList;
    }

    /**
     * returns the SelectionRecord vector python list of strings.
     */
    PyObject *_getSelections() {

        const std::vector<rr::SelectionRecord>& selections = ($self)->getSelections();

        unsigned size = selections.size();

        PyObject *pysel = PyList_New(size);

        unsigned j = 0;
        for (std::vector<rr::SelectionRecord>::const_iterator i = selections.begin();
             i != selections.end(); ++i) {
            std::string str = i->to_string();

            PyObject *pystr = PyString_FromString(str.c_str());
            PyList_SET_ITEM(pysel, j++, pystr);
        }

        return pysel;
    }

    /**
     * returns the SelectionRecord vector python list of strings.
     */
    PyObject *_getSteadyStateSelections() {

        const std::vector<rr::SelectionRecord>& selections = ($self)->getSteadyStateSelections();

        unsigned size = selections.size();

        PyObject *pysel = PyList_New(size);

        unsigned j = 0;
        for (std::vector<rr::SelectionRecord>::const_iterator i = selections.begin();
             i != selections.end(); ++i) {
            std::string str = i->to_string();

            PyObject *pystr = PyString_FromString(str.c_str());
            PyList_SET_ITEM(pysel, j++, pystr);
        }

        return pysel;
    }



   %pythoncode %{
        def getModel(self):
            return self._getModel()

        def _setConservedMoietyAnalysisProxy(self, value):
            self._setConservedMoietyAnalysis(value)
            self._makeProperties()

        __swig_getmethods__["selections"] = _getSelections # DEPRECATED
        __swig_setmethods__["selections"] = _setSelections # DEPRECATED
        __swig_getmethods__["timeCourseSelections"] = _getSelections
        __swig_setmethods__["timeCourseSelections"] = _setSelections
        __swig_getmethods__["steadyStateSelections"] = _getSteadyStateSelections
        __swig_setmethods__["steadyStateSelections"] = _setSteadyStateSelections
        __swig_getmethods__["conservedMoietyAnalysis"] = _getConservedMoietyAnalysis
        __swig_setmethods__["conservedMoietyAnalysis"] = _setConservedMoietyAnalysisProxy
        __swig_getmethods__["model"] = _getModel
        __swig_getmethods__["integrator"] = getIntegrator
        __swig_setmethods__["integrator"] = setIntegrator

        if _newclass:
            selections = property(_getSelections, _setSelections)
            timeCourseSelections = property(_getSelections, _setSelections)
            steadyStateSelections = property(_getSteadyStateSelections, _setSteadyStateSelections)
            conservedMoietyAnalysis=property(_getConservedMoietyAnalysis, _setConservedMoietyAnalysis)
            model = property(getModel)
            integrator = property(getIntegrator)


        # static list of properties added to the RoadRunner
        # class object
        _properties = []

        def _makeProperties(self):

            #global _properties

            # always clear the old properties
            for s in RoadRunner._properties:
                if s in RoadRunner.__swig_getmethods__:
                    del RoadRunner.__swig_getmethods__[s]
                if s in RoadRunner.__swig_setmethods__:
                    del RoadRunner.__swig_setmethods__[s]
                if hasattr(RoadRunner, s):
                    delattr(RoadRunner, s)

            # properties now empty
            RoadRunner._properties = []

            # check if we should make new properties
            if Config.getValue(Config.ROADRUNNER_DISABLE_PYTHON_DYNAMIC_PROPERTIES):
                return

            # can't make properties without a model.
            if self.getModel() is None:
                return

            def mk_fget(sel): return lambda self: self.getModel().__getitem__(sel)
            def mk_fset(sel): return lambda self, val: self.getModel().__setitem__(sel, val)


            def makeProperty(name, sel):
                fget = mk_fget(sel)
                fset = mk_fset(sel)
                RoadRunner.__swig_getmethods__[name] = fget
                RoadRunner.__swig_setmethods__[name] = fset
                setattr(RoadRunner, name, property(fget, fset))
                RoadRunner._properties.append(name)

            model = self.getModel()
            for s in model.getFloatingSpeciesIds():
                makeProperty(s, "[" + s + "]")  # concentrations
                makeProperty(s + "_amt", s)     # amounts


            for s in model.getBoundarySpeciesIds():
                makeProperty(s, "[" + s + "]")  # concentrations
                makeProperty(s + "_amt", s)     # amounts


            for s in model.getGlobalParameterIds() + model.getCompartmentIds() + model.getReactionIds() + model.getConservedMoietyIds():
                makeProperty(s, s)



        # Set up the python dyanic properties for model access,
        # save the original init method
        _swig_init = __init__

        def _new_init(self, *args):
            # if called with https, use Python for transport
            if len(args) >= 1:
                p = args[0]
                if hasattr(p,'startswith') and p.startswith('https://'):
                    try:
                        # Python3
                        from urllib.request import urlopen
                    except ImportError:
                        # Python2
                        from urllib2 import urlopen
                    sbml = urlopen(p).read()
                    try:
                        sbml = str(sbml.decode())
                    except:
                        pass
                    RoadRunner._swig_init(self, sbml)
                    RoadRunner._makeProperties(self)
                    return
            # Otherwise, use regular init
            RoadRunner._swig_init(self, *args)
            RoadRunner._makeProperties(self)

        # set the ctor to use the new init
        __init__ = _new_init




        def load(self, *args):
            self._load(*args)
            RoadRunner._makeProperties(self)


        def keys(self, types=_roadrunner.SelectionRecord_ALL):
            return self.getIds(types)

        def values(self, types=_roadrunner.SelectionRecord_ALL):
            return [self.getValue(k) for k in self.keys(types)]

        def items(self, types=_roadrunner.SelectionRecord_ALL):
            return [(k, self.getValue(k)) for k in self.keys(types)]

        def __len__(self):
            return len(self.keys())

        def iteritems(self, types=_roadrunner.SelectionRecord_ALL):
            """
            return an iterator over (key, value) pairs
            """
            return self.items(types).__iter__()

        def iterkeys(self, types=_roadrunner.SelectionRecord_ALL):
            """
            return an iterator over the mapping's keys
            """
            return self.keys(types).__iter__()

        def itervalues(self, types=_roadrunner.SelectionRecord_ALL):
            """
            return an iterator over the mapping's values
            """
            return self.values(types).__iter__()

        def simulate(self, start=None, end=None, points=None, selections=None, steps=None):
            '''
            Simulate the current SBML model.

            simulate accepts a up to four positional arguments. The first four (optional) arguments are treated as:

            1: start (the simulation starting time)

            2: end (the simulation end time)

            3: steps (the number of output points)

            4: List of Selections.

            All four of the positional arguments are optional. If any of the positional arguments are
            supplied as a list of strings, then they are interpreted as a list of selections.


            There is only one correct way to call simulate. If one positional argument is specified,
            it is the start time. If two are specified, they are the start and end time.
            The third argument is the number of output points!
            The fourth argument, if it is supplied, must be a list of strings that correspond to
            proper timecourse selections as in timeCourseSelections.
            The fifth argument, if supplied via keyword, is the number of intervals, not the
            number of points. Specifying intervals and points is an error.
            '''

            # check for errors
            import collections
            import sys
            if selections is not None:
                # check that selections is a sequence
                if not isinstance(selections, collections.Sequence):
                    raise ValueError('Expected a sequence type for selections')

                # check that selections contains only strings
                for x in selections:
                    if sys.version_info >= (3,0):
                        if not isinstance(x, str):
                            raise ValueError('Expected selections to be a sequence of strings')
                    else:
                        if not isinstance(x, basestring):
                            raise ValueError('Expected selections to be a sequence of strings')

            if points is not None and steps is not None:
                raise ValueError('Cannot specify both points and steps in simulate call')

            if points is not None:
                if points < 2:
                    raise ValueError('Number of points cannot be less than 2')

            if steps is not None:
                if steps < 1:
                    raise ValueError('Number of steps cannot be less than 1')

            # end error checking

            o = self.__simulateOptions
            originalSteps = o.steps

            if self.getIntegrator().hasValue('variable_step_size'):
                if self.getIntegrator().getValue('variable_step_size') == True:
                    o.steps = 0

            if start is not None:
                o.start = start

            if end is not None:
                o.end = end

            if points is not None:
                o.steps = points - 1

            if selections is not None:
                self.timeCourseSelections = selections

            if steps is not None:
                o.steps = steps

            result = self._simulate(o)

            o.steps = originalSteps

            return result

        def __simulateOld(self, *args, **kwargs):
            """
            DEPRECATED!!!!!!!!!!!!!!!!
            WILL BE REMOVED

            Simulate the current SBML model.

            simulate accepts a up to four positional arguments. The first four (optional) arguments are treated as:

            1: start (the simulation starting time)

            2: end (the simulation end time)

            3: steps (the number of output points)

            4: List of Selections.

            All four of the positional arguments are optional. If any of the positional arguments are
            supplied as a list of strings, then they are interpreted as a list of selections.


            There is only one correct way to call simulate. If one positional argument is specified,
            it is the start time. If two are specified, they are the start and end time.
            The third argument is the number of output points!
            The fourth argument, if it is supplied, must be a list of strings that correspond to
            proper timecourse selections as in timeCourseSelections.
            The fifth argument, if supplied via keyword, is the number of intervals, not the
            number of points. Specifying intervals and points is an error.

            Do NOT pass a `SimulateOptions` object to this function. SimulateOptions is DEPRECATED.

            Keyword arguments:

            integrator
                DEPRECATED: use setIntegrator method

            sel or selections
                A list of strings specifying what values to display in the output.

            plot
                DEPRECATED: use plot method.

            end
                The simulation end time. Note, setting the end time automatically sets
                the duration accordingly and visa versa.

            resetModel
                DEPRECATED

            reset
                DEPRECATED: use reset method.

            start
                The start time of the simulation time-series data. Often this is 0,
                but not necessarily.

            steps
                The number of steps at which the output is sampled. The samples are evenly spaced.
                When a simulation system calculates the data points to record, it will typically
                divide the duration by the number of time steps. Thus, for N steps, the output
                will have N+1 data rows.

            stiff
                DEPRECATED: use solver API (this setting only available for some solvers).

            seed
                DEPRECATED: use solver API (this setting only available for some solvers).


            :returns: a numpy array with each selected output time series being a
             column vector, and the 0'th column is the simulation time (if time is selected as an
             output).
            :rtype: numpy.ndarray
            """

            doPlot = False
            showPlot = True

            # cleanup tasks
            post_tasks = []

            # user specified number of steps via 3rd arg or steps=xxx
            haveSteps = False

            # variableStep = True was specified in args
            haveVariableStep = False
            o = self.__simulateOptions

            def steps_restore(v):
                def f():
                    o.steps = v
                return f
            post_tasks.append(steps_restore(o.steps))

            o.steps = 50
            if self.getIntegrator().hasValue('variable_step_size'):
                if self.getIntegrator().getValue('variable_step_size') == True:
                    o.steps = 0
            stepsSpecified = False

            # did the options originally have a seed, if so, don't delete it when we're done
            hadSeed = "seed" in o

            # check if we have just a sim options
            if len(args) >= 1:
                if type(args[0]) == type(self.__simulateOptions):
                    o = args[0]
                elif type(args[0]) == list:
                    # its a selection list
                    self.timeCourseSelections = args[0]
                elif isinstance(args[0], (int, float)):
                    # treat it as a number
                    o.start = args[0]
                else:
                    raise ValueError("argument 1 must be either a number, list or "
                                     "SimulateOptions object, recieved: {0}".format(str(args[0])))

            # second arg is treated as sim end time
            if len(args) >= 2:
                if type(args[1]) == list:
                    # its a selection list
                    self.timeCourseSelections = args[1]
                elif isinstance(args[1], (int, float)):
                    # treat it as a number
                    o.end = args[1]
                else:
                    raise ValueError("argument 2 must be either a number, list or "
                                     "SimulateOptions object, recieved: {0}".format(str(args[1])))


            # third arg is treated as number of points
            if len(args) >= 3:
                if type(args[2]) == list:
                    # its a selection list
                    self.timeCourseSelections = args[2]
                elif isinstance(args[2], (int, float)):
                    # treat it as a number
                    o.steps = args[2]-1
                    stepsSpecified = True
                    if o.steps < 1:
                      raise RuntimeError('Number of points must be 2 or more')
                    haveSteps = True
                else:
                    raise ValueError("argument 3 must be either a number, list or "
                                     "SimulateOptions object, recieved: {0}".format(str(args[2])))

            # forth arg may be a list (currently)
            if len(args) >= 4:
                if type(args[3]) == list:
                    # its a selection list
                    self.timeCourseSelections = args[3]
                else:
                    raise ValueError("argument 4 (if given) must be a list of timeCourseSelections "
                                     ", recieved: {0}".format(str(args[3])))


            # go through the list of keyword args
            for k,v in kwargs.items():

                # changing integrators.
                if k == "integrator":
                    if type(v) == str:
                        # this automatically sets the variable / fixed time step
                        # according to integrator type, raises exception if invalid
                        # integrator string.
                        self.setIntegrator(v)
                    else:
                        raise Exception("{0} is invalid argument for integrator, integrator name must be a string.".format(v))
                    continue

                # specifying timeCourseSelections:
                if k == "timeCourseSelections" or k == "sel":
                    self.timeCourseSelections = v
                    continue

                if k == "steps":
                    o.steps = v
                    stepsSpecified = True
                    continue

                if k == "start":
                    o.start = v
                    continue

                if k == "end":
                    o.end = v
                    continue

                # reset model, also accept 'reset'
                if k == "reset" or k == "resetModel":
                    o.resetModel = v
                    continue

                # check if variableStep was explicitly specified, this overrides the steps
                # positional arg
                if k == "variableStep":
                    raise KeyError('Do NOT pass variableStep to simulate. Use integrator API: r.getIntegrator().setValue("variable_step_size", True)')
                    haveVariableStep = True
                    self.getIntegrator().setValue('variable_step_size', v)
                    if not stepsSpecified:
                        o.steps = 0
                    continue

                if k == "plot":
                    raise RuntimeError('plot argument is deprecated, use plot method')
                    continue

                if k == "show":
                    showPlot = v
                    continue

                if k == "stiff" and self.getIntegrator().hasValue('stiff'):
                    raise KeyError('Do NOT pass stiff to simulate. Use the integrator API: r.getIntegrator().setValue("stiff", True)')
                    def stiff_restore(v):
                        def f():
                            self.getIntegrator().setValue('stiff', v)
                        return f
                    self.getIntegrator().setValue('stiff', kwargs[k])
                    post_tasks.append(stiff_restore(self.getIntegrator().getValue('stiff')))
                    continue

                # if its not one of these, just set the item on the dict, and
                # if the inegrator cares about it, it will use it.
                # if its one of these, set it.
                raise KeyError('No such argument: {}'.format(k))



            # if we are doing a stochastic sim,
            # explicit options of variableStep trumps everything,
            # if not explicit, variableStep is if number of steps was specified,
            # if no steps, varStep = true, false otherwise.
            if self.getIntegrator().getIntegrationMethod() == \
                Integrator.Stochastic and not haveVariableStep:
                self.getIntegrator().setValue('variable_step_size', not haveSteps)

            # the options are set up, now actually run the simuation...
            result = self._simulate(o)

            if not hadSeed:
                del o["seed"]

            if doPlot:
                self.plot(result=None, loc='upper left', show=showPlot)

            # revert any settings we changed
            for x in post_tasks:
                x()

            return result

        # ---------------------------------------------------------------------
        # Reset Methods
        # ---------------------------------------------------------------------
        def resetToOrigin(self):
            """ Reset model to state when first loaded.

            This resets the model back to the state when it was FIRST loaded,
            this includes all init() and parameters such as k1 etc.

            identical to:
                r.reset(SelectionRecord.ALL)
            """
            self.reset(SelectionRecord.ALL)

        def resetAll(self):
            """ Reset all model variables to CURRENT init(X) values.

            This resets all variables, S1, S2 etc to the CURRENT init(X) values. It also resets all
            parameters back to the values they had when the model was first loaded.
            """
            self.reset(SelectionRecord.TIME |
                       SelectionRecord.RATE |
                       SelectionRecord.FLOATING |
                       SelectionRecord.GLOBAL_PARAMETER)


        def getAvailableIntegrators(self):
            """
            get a list of available integrator names.
            """
            return self.getExistingIntegratorNames()


        def plot(self, result=None, loc='upper left', show=True):
            """
            RoadRunner.plot([show])

            Plot the previously run simulation result using Matplotlib.

            This takes the contents of the simulation result and builds a
            legend from the selection list.


            If the optional prameter 'show' [default is True] is given, the pylab
            show() method is called.
            """

            try:
                import matplotlib.pyplot as p
            except ImportError:
                raise ImportError('Could not import matplotlib - please install matplotlib to enable plotting functionality')

            result = self.getSimulationData()

            if result is None:
                raise Exception("no simulation result")

            rval = []

            # check if standard numpy array
            if result.dtype.names is None:

                timeCourseSelections = self.timeCourseSelections

                if len(result.shape) != 2 or result.shape[1] != len(timeCourseSelections):
                    raise Exception("simulation result columns not equal to number of selections, likely a simulation has not been run")

                times = result[:,0]

                for i in range(1, len(timeCourseSelections)):
                    series = result[:,i]
                    name = timeCourseSelections[i]
                    rval.append(p.plot(times, series, label=str(name), linewidth=2))

            # result is structured array
            else:
                if len(result.dtype.names) < 1:
                    raise Exception('no columns to plot')

                time = result.dtype.names[0]

                for name in result.dtype.names[1:]:
                    rval.append(p.plot(result[time], result[name], label=name, linewidth=2))

            if show:
                p.show()

            return rval

        def plotLegend(self):
            try:
                import matplotlib.pyplot as p
            except ImportError:
                raise ImportError('Could not import matplotlib - please install matplotlib to enable plotting functionality')
            p.legend()

        def showPlot(self):
            try:
                import matplotlib.pyplot as p
            except ImportError:
                raise ImportError('Could not import matplotlib - please install matplotlib to enable plotting functionality')
            p.show()

        def getIndependentFloatingSpeciesIds(self):
            return list(self._getIndependentFloatingSpeciesIds())

        def getDependentFloatingSpeciesIds(self):
            return list(self._getDependentFloatingSpeciesIds())

        def getFloatingSpeciesAmountRates(self):
            return self.model.getFloatingSpeciesAmountRates()

        def getReactionRates(self):
            return self.getModel().getReactionRates()

        if _newclass:
            integrator = property(getIntegrator, setIntegrator)

        def setIntegratorSetting(self, integratorName, settingName, value):
            import sys
            if sys.version_info >= (3,0):
                if not isinstance(integratorName, str):
                    raise ValueError('Expected integratorName to be a string')
                if not isinstance(settingName, str):
                    raise ValueError('Expected settingName to be a string')
            else:
                if not isinstance(integratorName, basestring):
                    raise ValueError('Expected integratorName to be a string')
                if not isinstance(settingName, basestring):
                    raise ValueError('Expected settingName to be a string')

            # store original integrator and switch back to it afterwards
            origIntegrator = self.getIntegrator().getName()

            self.setIntegrator(integratorName)
            self.getIntegrator().setValue(settingName, value)
            self.setIntegrator(origIntegrator)

        @property
        def steadyStateSolver(self):
            '''The current steady state solver'''
            return self.getSteadyStateSolver()

        @steadyStateSolver.setter
        def steadyStateSolver(self, v):
            self.setSteadyStateSolver(v)

        def _diffstep_getter(self):
            '''Differential step size used in MCA'''
            return self.getDiffStepSize()

        def _diffstep_stter(self, v):
            print('diffstep.setter')
            self.setDiffStepSize(v)

        __swig_getmethods__['diffstep'] = _diffstep_getter
        __swig_setmethods__['diffstep'] = _diffstep_stter

        if _newclass:
            diffstep = property(_diffstep_getter, _diffstep_stter)

        def _steadyStateThresh_getter(self):
            '''Steady state threshold used in MCA'''
            return self.getSteadyStateThreshold()

        def _steadyStateThresh_setter(self, v):
            self.setSteadyStateThreshold(v)

        __swig_getmethods__['steadyStateThresh'] = _steadyStateThresh_getter
        __swig_setmethods__['steadyStateThresh'] = _steadyStateThresh_setter

        if _newclass:
            steadyStateThresh = property(_steadyStateThresh_getter, _steadyStateThresh_setter)
    %}
}

%{
    rr::SimulateOptions* rr_RoadRunner___simulateOptions_get(RoadRunner* r) {
        //Log(Logger::LOG_WARNING) << "DO NOT USE simulateOptions, it is DEPRECATED";
        return &r->getSimulateOptions();
    }

    void rr_RoadRunner___simulateOptions_set(RoadRunner* r, const rr::SimulateOptions* opt) {
        //Log(Logger::LOG_WARNING) << "DO NOT USE simulateOptions, it is DEPRECATED";
        r->setSimulateOptions(*opt);
    }


    rr::RoadRunnerOptions* rr_RoadRunner_options_get(RoadRunner* r) {
        Log(Logger::LOG_WARNING) << "DO NOT USE options, it is DEPRECATED";
        return &r->getOptions();
    }

    void rr_RoadRunner_options_set(RoadRunner* r, const rr::RoadRunnerOptions* opt) {
        Log(Logger::LOG_WARNING) << "DO NOT USE options, it is DEPRECATED";
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
    bool structuredResult;
    bool copyResult;

    std::string __repr__() {
        return ($self)->toRepr();
    }


    std::string __str__() {
        return ($self)->toString();
    }

    /**
     * makes a copy of this object.
     * Python normally just keeps references to objects, and this forces a true
     * copy. Note, we heed to add the SWIG_POINTER_OWN to the function below
     * so that when the returned object is destroyed (by Python), the C++
     * object will also be deleted.
     */
    PyObject *copy() {
        SimulateOptions *pThis = $self;
        SimulateOptions *other = new SimulateOptions(*pThis);
        return SWIG_NewPointerObj(SWIG_as_voidptr(other), SWIGTYPE_p_rr__SimulateOptions, SWIG_POINTER_OWN );
    }

    %pythoncode %{
        def getListener(self):
            return self._getListener()

        def setListener(self, listener):
            if listener is None:
                self._clearListener()
            else:
                self._setListener(listener)
    %}

}

%{
    double rr_SimulateOptions_end_get(SimulateOptions* opt) {
        return opt->start + opt->duration;
    }

    void rr_SimulateOptions_end_set(SimulateOptions* opt, double end) {
        opt->duration = end - opt->start;
    }

    bool rr_SimulateOptions_structuredResult_get(SimulateOptions* opt) {
        return opt->structured_result;
    }

    void rr_SimulateOptions_structuredResult_set(SimulateOptions* opt, bool value) {
        opt->structured_result = value;
    }

    bool rr_SimulateOptions_copyResult_get(SimulateOptions* opt) {
        return opt->getItem("copyResult");
    }

    void rr_SimulateOptions_copyResult_set(SimulateOptions* opt, bool value) {
        opt->setItem("copyResult", value);
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

    /**
     * creates a function signature of
     * SWIGINTERN PyObject *rr_ExecutableModel_getIds(rr::ExecutableModel *self,int types);
     */
    PyObject *getIds(int types) {
        std::list<std::string> ids;

        ($self)->getIds(types, ids);

        unsigned size = ids.size();

        PyObject* pyList = PyList_New(size);

        unsigned j = 0;

        for (std::list<std::string>::const_iterator i = ids.begin(); i != ids.end(); ++i)
        {
            const std::string& id  = *i;
            PyObject* pyStr = PyString_FromString(id.c_str());
            PyList_SET_ITEM(pyList, j++, pyStr);
        }

        return pyList;
    }

    /***
     ** get values section
     ***/

    PyObject *getFloatingSpeciesAmounts(int len, int const *indx) {
        return _ExecutableModel_getValues($self, &rr::ExecutableModel::getFloatingSpeciesAmounts,
                                          &rr::ExecutableModel::getNumFloatingSpecies, len, indx);
    }

    PyObject *getFloatingSpeciesAmounts() {
        return _ExecutableModel_getValues($self, &rr::ExecutableModel::getFloatingSpeciesAmounts,
                                          &rr::ExecutableModel::getNumFloatingSpecies, (int)0, (int const*)0);
    }

    PyObject *getFloatingSpeciesAmountRates(int len, int const *indx) {
        return _ExecutableModel_getValues($self, &rr::ExecutableModel::getFloatingSpeciesAmountRates,
                                         &rr::ExecutableModel::getNumIndFloatingSpecies,  len, indx);
    }

    PyObject *getFloatingSpeciesAmountRates() {
        return _ExecutableModel_getValues($self, &rr::ExecutableModel::getFloatingSpeciesAmountRates,
                                          &rr::ExecutableModel::getNumIndFloatingSpecies, (int)0, (int const*)0);
    }

    PyObject *getFloatingSpeciesConcentrationRates(int len, int const *indx) {
        return _ExecutableModel_getValues($self, &rr::ExecutableModel::getFloatingSpeciesConcentrationRates,
                                         &rr::ExecutableModel::getNumIndFloatingSpecies,  len, indx);
    }

    PyObject *getFloatingSpeciesConcentrationRates() {
        return _ExecutableModel_getValues($self, &rr::ExecutableModel::getFloatingSpeciesConcentrationRates,
                                          &rr::ExecutableModel::getNumIndFloatingSpecies, (int)0, (int const*)0);
    }

    PyObject *getFloatingSpeciesConcentrations(int len, int const *indx) {
        return _ExecutableModel_getValues($self, &rr::ExecutableModel::getFloatingSpeciesConcentrations,
                                         &rr::ExecutableModel::getNumFloatingSpecies,  len, indx);
    }

    PyObject *getFloatingSpeciesConcentrations() {
        return _ExecutableModel_getValues($self, &rr::ExecutableModel::getFloatingSpeciesConcentrations,
                                          &rr::ExecutableModel::getNumFloatingSpecies, (int)0, (int const*)0);
    }

    PyObject *getBoundarySpeciesAmounts(int len, int const *indx) {
        return _ExecutableModel_getValues($self, &rr::ExecutableModel::getBoundarySpeciesAmounts,
                                         &rr::ExecutableModel::getNumBoundarySpecies,  len, indx);
    }

    PyObject *getBoundarySpeciesAmounts() {
        return _ExecutableModel_getValues($self, &rr::ExecutableModel::getBoundarySpeciesAmounts,
                                          &rr::ExecutableModel::getNumBoundarySpecies, (int)0, (int const*)0);
    }

    PyObject *getBoundarySpeciesConcentrations(int len, int const *indx) {
        return _ExecutableModel_getValues($self, &rr::ExecutableModel::getBoundarySpeciesConcentrations,
                                         &rr::ExecutableModel::getNumBoundarySpecies,  len, indx);
    }

    PyObject *getBoundarySpeciesConcentrations() {
        return _ExecutableModel_getValues($self, &rr::ExecutableModel::getBoundarySpeciesConcentrations,
                                          &rr::ExecutableModel::getNumBoundarySpecies, (int)0, (int const*)0);
    }
    PyObject *getGlobalParameterValues(int len, int const *indx) {
        return _ExecutableModel_getValues($self, &rr::ExecutableModel::getGlobalParameterValues,
                                         &rr::ExecutableModel::getNumGlobalParameters,  len, indx);
    }

    PyObject *getGlobalParameterValues() {
        return _ExecutableModel_getValues($self, &rr::ExecutableModel::getGlobalParameterValues,
                                          &rr::ExecutableModel::getNumGlobalParameters, (int)0, (int const*)0);
    }

    PyObject *getCompartmentVolumes(int len, int const *indx) {
        return _ExecutableModel_getValues($self, &rr::ExecutableModel::getCompartmentVolumes,
                                         &rr::ExecutableModel::getNumCompartments,  len, indx);
    }

    PyObject *getCompartmentVolumes() {
        return _ExecutableModel_getValues($self, &rr::ExecutableModel::getCompartmentVolumes,
                                          &rr::ExecutableModel::getNumCompartments, (int)0, (int const*)0);
    }

    PyObject *getConservedMoietyValues(int len, int const *indx) {
        return _ExecutableModel_getValues($self, &rr::ExecutableModel::getConservedMoietyValues,
                                         &rr::ExecutableModel::getNumConservedMoieties,  len, indx);
    }

    PyObject *getConservedMoietyValues() {
        return _ExecutableModel_getValues($self, &rr::ExecutableModel::getConservedMoietyValues,
                                          &rr::ExecutableModel::getNumConservedMoieties, (int)0, (int const*)0);
    }

    PyObject *getReactionRates(int len, int const *indx) {
        return _ExecutableModel_getValues($self, &rr::ExecutableModel::getReactionRates,
                                         &rr::ExecutableModel::getNumReactions,  len, indx);
    }

    PyObject *getReactionRates() {
        return _ExecutableModel_getValues($self, &rr::ExecutableModel::getReactionRates,
                                          &rr::ExecutableModel::getNumReactions, (int)0, (int const*)0);
    }

    PyObject *getFloatingSpeciesInitConcentrations() {
        return _ExecutableModel_getValues($self, &rr::ExecutableModel::getFloatingSpeciesInitConcentrations,
                                          &rr::ExecutableModel::getNumFloatingSpecies, (int)0, (int const*)0);
    }

    PyObject *getFloatingSpeciesInitAmounts() {
        return _ExecutableModel_getValues($self, &rr::ExecutableModel::getFloatingSpeciesInitAmounts,
                                          &rr::ExecutableModel::getNumFloatingSpecies, (int)0, (int const*)0);
    }

    PyObject *getCompartmentInitVolumes() {
        return _ExecutableModel_getValues($self, &rr::ExecutableModel::getCompartmentInitVolumes,
                                          &rr::ExecutableModel::getNumCompartments, (int)0, (int const*)0);
    }

    /***
     ** state vector section
     ***/



    /**
     * overload which returns a copy of the state vector
     */
    PyObject *getStateVector() {
        int len = ($self)->getStateVector(0);

        npy_intp dims[1] = {len};
        PyObject *array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        VERIFY_PYARRAY(array);

        if (!array) {
            // TODO error handling.
            return 0;
        }

        double *data = (double*)PyArray_DATA((PyArrayObject*)array);

        ($self)->getStateVector(data);

        return array;
    }


    void getStateVector(int out_len, double* out_values) {
        int len = ($self)->getStateVector(0);

        if (len > out_len) {
            PyErr_Format(PyExc_ValueError,
                         "given array of size %d, insufficient size for state vector %d.",
                         out_len, len);
            return;
        }

        ($self)->getStateVector(out_values);
    }

    /**
     * get a copy of the state vector rate using the current state.
     */
    PyObject *getStateVectorRate(double time = NAN) {
        int len = ($self)->getStateVector(0);

        npy_intp dims[1] = {len};
        PyObject *array = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        VERIFY_PYARRAY(array);

        if (!array) {
            // TODO error handling.
            return 0;
        }

        if (isnan(time)) {
            time = ($self)->getTime();
        }

        double *data = (double*)PyArray_DATA((PyArrayObject*)array);

        ($self)->getStateVectorRate(time, 0, data);

        return array;
    }


    PyObject *getStateVectorRate(double time, PyObject *arg) {

        // length of state vector
        int len = ($self)->getStateVector(0);

        if (isnan(time)) {
            time = ($self)->getTime();
        }

        // check if the pyobj is an npy double array
        PyArrayObject *array  = obj_to_array_no_conversion(arg, NPY_DOUBLE);

        // need contigous and native
        // these set py error if they fail
        if (array && require_contiguous(array) && require_native(array))
        {
            // The number of dimensions in the array.
            int ndim = PyArray_NDIM(array);

            // pointer to the dimensions/shape of the array.
            npy_intp *pdims = PyArray_DIMS(array);

            if (ndim > 0 && pdims[ndim-1] == len) {

                // total number of elements in array
                npy_intp size = PyArray_SIZE(array);

                printf("size: %i\n", (int)size);

                // how many state vectors we have in given array
                int nvec = size / len;

                // pointer to start of data block
                double* stateVec = (double*)PyArray_DATA(array);

                // make result data, copy descriptor
                PyArray_Descr *dtype = PyArray_DESCR(array);
                Py_INCREF(dtype);

                // If strides is NULL, then the array strides are computed as
                // C-style contiguous (default)
                PyObject *result = PyArray_NewFromDescr(&PyArray_Type,
                                                        dtype,
                                                        ndim,
                                                        pdims,
                                                        NULL,
                                                        NULL,
                                                        0,
                                                        NULL);

                // out data
                double* stateVecRate = (double*)PyArray_DATA(result);

                // get the state vector rates, bump the data pointers
                // to the next state vec
                for (int i = 0; i < nvec; ++i) {
                    ($self)->getStateVectorRate(time, stateVec, stateVecRate);
                    stateVec += len;
                    stateVecRate += len;
                }

                return result;
            }

            PyErr_Format(PyExc_TypeError,
                         "Require an N dimensional array where N must be at least 1 and "
                         "the trailing dimension must be the same as the state vector size. "
                         "require trailing dimension of %i but recieved %d", len, (int)pdims[ndim-1]);


        }

        // error case, PyErr is set if we get here.
        return 0;
    }


    PyObject *getStateVectorRate(double time, PyObject *arg1, PyObject *arg2) {

        // length of state vector
        int len = ($self)->getStateVector(0);

        if (isnan(time)) {
            time = ($self)->getTime();
        }

        // check if the pyobj is an npy double array
        PyArrayObject *array1  = obj_to_array_no_conversion(arg1, NPY_DOUBLE);
        PyArrayObject *array2  = obj_to_array_no_conversion(arg2, NPY_DOUBLE);

        // need contigous and native
        // these set py error if they fail
        if (array1 && require_contiguous(array1) && require_native(array1) &&
            array2 && require_contiguous(array2) && require_native(array2))
        {
            // The number of dimensions in the array.
            int ndim1 = PyArray_NDIM(array1);
            int ndim2 = PyArray_NDIM(array2);

            // pointer to the dimensions/shape of the array.
            npy_intp *pdims1 = PyArray_DIMS(array1);
            npy_intp *pdims2 = PyArray_DIMS(array2);

            if (ndim1 > 0 && pdims1[ndim1-1] == len &&
                ndim2 > 0 && pdims2[ndim2-1] == len ) {

                // total number of elements in array
                npy_intp size = PyArray_SIZE(array1);

                if (size == PyArray_SIZE(array2)) {

                    printf("size: %i\n", (int)size);

                    // how many state vectors we have in given array
                    int nvec = size / len;

                    // pointer to start of data block
                    double* stateVec = (double*)PyArray_DATA(array1);

                    // out data
                    double* stateVecRate = (double*)PyArray_DATA(array2);

                    // get the state vector rates, bump the data pointers
                    // to the next state vec
                    for (int i = 0; i < nvec; ++i) {
                        ($self)->getStateVectorRate(time, stateVec, stateVecRate);
                        stateVec += len;
                        stateVecRate += len;
                    }

                    // only error free result case
                    // caller should decref the array so we need incrref it.
                    Py_INCREF(arg2);
                    return arg2;

                } else {
                    PyErr_Format(PyExc_TypeError,
                                 "Both arrays must be the same size and shape, array 1 size: %i, "
                                 "array 2 size: %i", (int)size, (int)PyArray_SIZE(array2));
                }
            } else {
                PyErr_Format(PyExc_TypeError,
                             "Require an N dimensional array where N must be at least 1 and "
                             "the trailing dimension must be the same as the state vector size. "
                             "require trailing dimension of %i but recieved %d", len, (int)pdims1[ndim1-1]);
            }
        }

        // error case, PyErr is set if we get here.
        return 0;
    }




    /***
     ** get ids section
     ***/


    PyObject *getFloatingSpeciesIds() {
        return rr_ExecutableModel_getIds($self, rr::SelectionRecord::FLOATING_AMOUNT);
    }

    PyObject *getBoundarySpeciesIds() {
        return rr_ExecutableModel_getIds($self, rr::SelectionRecord::BOUNDARY_AMOUNT);
    }

    PyObject *getGlobalParameterIds() {
        return rr_ExecutableModel_getIds($self, rr::SelectionRecord::GLOBAL_PARAMETER);
    }

    PyObject *getCompartmentIds() {
        return rr_ExecutableModel_getIds($self, rr::SelectionRecord::COMPARTMENT);
    }

    PyObject *getConservedMoietyIds() {
        return rr_ExecutableModel_getIds($self, rr::SelectionRecord::CONSERVED_MOIETY);
    }

    PyObject *getReactionIds() {
        return rr_ExecutableModel_getIds($self, rr::SelectionRecord::REACTION_RATE);
    }

    PyObject *getFloatingSpeciesInitAmountIds() {
        return rr_ExecutableModel_getIds($self, rr::SelectionRecord::INITIAL_FLOATING_AMOUNT);
    }

    PyObject *getFloatingSpeciesInitConcentrationIds() {
        return rr_ExecutableModel_getIds($self, rr::SelectionRecord::INITIAL_FLOATING_CONCENTRATION);
    }

    PyObject *getFloatingSpeciesAmountRateIds() {
        return rr_ExecutableModel_getIds($self, rr::SelectionRecord::FLOATING_AMOUNT_RATE);
    }

    PyObject *getStateVectorIds() {
        return rr_ExecutableModel_getIds($self, rr::SelectionRecord::STATE_VECTOR);
    }

    PyObject *getEventIds() {
        return rr_ExecutableModel_getIds($self, rr::SelectionRecord::EVENT);
    }



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
            PyErr_Format(PyExc_ValueError,
                         "Arrays of lengths (%d,%d) given",
                         leni, lenv);
            return -1;
        }
        return $self->setFloatingSpeciesAmounts(leni, indx, values);
    }


    int setFloatingSpeciesConcentrations(int leni, int const* indx, int lenv, double const *values) {
        if (leni != lenv) {
            PyErr_Format(PyExc_ValueError,
                         "Arrays of lengths (%d,%d) given",
                         leni, lenv);
            return -1;
        }
        return $self->setFloatingSpeciesConcentrations(leni, indx, values);
    }

    int setBoundarySpeciesConcentrations(int leni, int const* indx, int lenv, double const *values) {
        if (leni != lenv) {
            PyErr_Format(PyExc_ValueError,
                         "Arrays of lengths (%d,%d) given",
                         leni, lenv);
            return -1;
        }
        return $self->setBoundarySpeciesConcentrations(leni, indx, values);
    }

    int setGlobalParameterValues(int leni, int const* indx, int lenv, double const *values) {
        if (leni != lenv) {
            PyErr_Format(PyExc_ValueError,
                         "Arrays of lengths (%d,%d) given",
                         leni, lenv);
            return -1;
        }
        return $self->setGlobalParameterValues(leni, indx, values);
    }

    int setCompartmentVolumes(int leni, int const* indx, int lenv, double const *values) {
        if (leni != lenv) {
            PyErr_Format(PyExc_ValueError,
                         "Arrays of lengths (%d,%d) given",
                         leni, lenv);
            return -1;
        }
        return $self->setCompartmentVolumes(leni, indx, values);
    }

    int setConservedMoietyValues(int leni, int const* indx, int lenv, double const *values) {
        if (leni != lenv) {
            PyErr_Format(PyExc_ValueError,
                         "Arrays of lengths (%d,%d) given",
                         leni, lenv);
            return -1;
        }
        return $self->setConservedMoietyValues(leni, indx, values);
    }

    int setFloatingSpeciesInitConcentrations(int leni, int const* indx, int lenv, double const *values) {
        if (leni != lenv) {
            PyErr_Format(PyExc_ValueError,
                         "Arrays of lengths (%d,%d) given",
                         leni, lenv);
            return -1;
        }
        return $self->setFloatingSpeciesInitConcentrations(leni, indx, values);
    }


    int setFloatingSpeciesInitAmounts(int leni, int const* indx, int lenv, double const *values) {
        if (leni != lenv) {
            PyErr_Format(PyExc_ValueError,
                         "Arrays of lengths (%d,%d) given",
                         leni, lenv);
            return -1;
        }
        return $self->setFloatingSpeciesInitAmounts(leni, indx, values);
    }


    int setCompartmentInitVolumes(int leni, int const* indx, int lenv, double const *values) {
        if (leni != lenv) {
            PyErr_Format(PyExc_ValueError,
                         "Arrays of lengths (%d,%d) given",
                         leni, lenv);
            return -1;
        }
        return $self->setCompartmentInitVolumes(leni, indx, values);
    }


    int setFloatingSpeciesInitConcentrations(int leni, int const* indx, int lenv, double const *values) {
        if (leni != lenv) {
            PyErr_Format(PyExc_ValueError,
                         "Arrays of lengths (%d,%d) given",
                         leni, lenv);
            return -1;
        }
        return $self->setFloatingSpeciesInitConcentrations(leni, indx, values);
    }

    int setFloatingSpeciesInitAmounts(int leni, int const* indx, int lenv, double const *values) {
        if (leni != lenv) {
            PyErr_Format(PyExc_ValueError,
                         "Arrays of lengths (%d,%d) given",
                         leni, lenv);
            return -1;
        }
        return $self->setFloatingSpeciesInitAmounts(leni, indx, values);
    }


    int setCompartmentInitVolumes(int leni, int const* indx, int lenv, double const *values) {
        if (leni != lenv) {
            PyErr_Format(PyExc_ValueError,
                         "Arrays of lengths (%d,%d) given",
                         leni, lenv);
            return -1;
        }
        return $self->setCompartmentInitVolumes(leni, indx, values);
    }


    PyObject* getCurrentStoichiometryMatrix() {
        int rows = 0;
        int cols = 0;
        double* data = 0;

        $self->getStoichiometryMatrix(&rows, &cols, &data);

        int nd = 2;
        npy_intp dims[2] = {rows, cols};

        PyObject *pArray = PyArray_New(&PyArray_Type, nd, dims, NPY_DOUBLE, NULL, data, 0,
                NPY_CARRAY | NPY_OWNDATA, NULL);
        VERIFY_PYARRAY(pArray);

        return pArray;
    }


    /**
     * get values.
     */

    double __getitem__(const std::string& id) {
        return ($self)->getValue(id);
    }

    void __setitem__(const std::string& id, double value) {
        ($self)->setValue(id, value);
    }


    std::string __repr__() {
        std::stringstream s;
        s << "<roadrunner.ExecutableModel() { this = " << (void*)$self << " }>";
        return s.str();
    }

    /**
     * events section
     *
     * TODO, the returned event is not valid after the model object is freed.
     * is this OK???
     */
    rr::PyEventListener *getEvent(int index) {
        ExecutableModel *p = $self;
        EventListenerPtr e = p->getEventListener(index);

        if(e) {
            PyEventListener *impl = dynamic_cast<PyEventListener*>(e.get());
            return impl;
        } else {
            PyEventListener *impl = new PyEventListener();
            p->setEventListener(index, EventListenerPtr(impl));
            return impl;
        }
    }

    rr::PyEventListener *getEvent(const std::string& eventId) {
        int index = ($self)->getEventIndex(eventId);

        if (index >= 0) {
            ExecutableModel *p = $self;
            EventListenerPtr e = p->getEventListener(index);

            if(e) {
                PyEventListener *impl = dynamic_cast<PyEventListener*>(e.get());
                return impl;
            } else {
                PyEventListener *impl = new PyEventListener();
                p->setEventListener(index, EventListenerPtr(impl));
                return impl;
            }

        } else {
            throw std::out_of_range(std::string("could not find index for event ") + eventId);
        }
    }

    %pythoncode %{

        def keys(self, types=_roadrunner.SelectionRecord_ALL):
            return self.getIds(types)

        def values(self, types=_roadrunner.SelectionRecord_ALL):
            return [self.getValue(k) for k in self.keys(types)]

        def items(self, types=_roadrunner.SelectionRecord_ALL):
            return [(k, self.getValue(k)) for k in self.keys(types)]

        def __len__(self):
            return len(self.keys())

        def iteritems(self, types=_roadrunner.SelectionRecord_ALL):
            """
            return an iterator over (key, value) pairs
            """
            return self.items(types).__iter__()

        def iterkeys(self, types=_roadrunner.SelectionRecord_ALL):
            """
            return an iterator over the mapping's keys
            """
            return self.keys(types).__iter__()

        def itervalues(self, types=_roadrunner.SelectionRecord_ALL):
            """
            return an iterator over the mapping's values
            """
            return self.values(types).__iter__()
    %}
}

%extend rr::Logger {
    static void enablePythonLogging() {
        PyLoggerStream::enablePythonLogging();
    }

    static void disablePythonLogging() {
        PyLoggerStream::disablePythonLogging();
    }
}

%extend rr::Solver {
    %pythoncode %{
        def __dir__(self):
            x = dir(type(self))
            x += self.getSettings()
            return x

        def __getattr__(self, name):
            if(name in self.getSettings()):
                return Solver.getValue(self, name)
            else:
                return _swig_getattr(self, Integrator, name)

        def __setattr__(self, name, value):
            if(name != 'this' and name in self.getSettings()):
                self.setValue(name, value)
            else:
                _swig_setattr(self, Integrator, name, value)

        def getSetting(self, k):
            return self.getValue(k)

        def setSetting(self, k, v):
            return self.setValue(k, v)
    %}
}

%extend rr::Integrator {

    void _setListener(const rr::PyIntegratorListenerPtr &listener) {

        Log(rr::Logger::LOG_INFORMATION) << __FUNC__ << ", use count: " << listener.use_count();

        cxx11_ns::shared_ptr<rr::IntegratorListener> i =
            cxx11_ns::dynamic_pointer_cast<rr::IntegratorListener>(listener);

        Log(rr::Logger::LOG_INFORMATION) << __FUNC__ << ", after cast use count: " << listener.use_count();

        ($self)->setListener(i);
    }

    rr::PyIntegratorListenerPtr _getListener() {

        Log(rr::Logger::LOG_INFORMATION) << __FUNC__;

        rr::IntegratorListenerPtr l = ($self)->getListener();

        rr::PyIntegratorListenerPtr ptr =
            cxx11_ns::dynamic_pointer_cast<rr::PyIntegratorListener>(l);

        Log(rr::Logger::LOG_INFORMATION) << __FUNC__ << ", use count: " << ptr.use_count();

        return ptr;
    }

    void _clearListener() {
        rr::IntegratorListenerPtr current = ($self)->getListener();

        Log(rr::Logger::LOG_INFORMATION) << __FUNC__ << ", current use count before clear: " << current.use_count();

        ($self)->setListener(rr::IntegratorListenerPtr());

        Log(rr::Logger::LOG_INFORMATION) << __FUNC__ << ", current use count after clear: " << current.use_count();
    }

    // we want to get the listener back as a PyIntegratorListener, however
    // swig won't let us ignore by return value and if we ignore getListener,
    // it ignores any extended version. So, we have to make an extended
    // _getListener() above, and call it from python like this.
    %pythoncode %{
        def getListener(self):
            return self._getListener()

        def setListener(self, listener):
            if listener is None:
                self._clearListener()
            else:
                self._setListener(listener)

        __swig_getmethods__["listener"] = getListener
        __swig_setmethods__["listener"] = setListener
        if _newclass:
            listener = property(getListener, setListener)

        def __dir__(self):
            x = dir(type(self))
            x += self.getSettings()
            return x

        def __getattr__(self, name):
            if(name in self.getSettings()):
                return Solver.getValue(self, name)
            else:
                return _swig_getattr(self, Integrator, name)

        def __setattr__(self, name, value):
            if(name != 'this' and name in self.getSettings()):
                self.setValue(name, value)
            else:
                _swig_setattr(self, Integrator, name, value)

        def __repr__(self):
            return self.toRepr()

        def getSetting(self, k):
            return self.getValue(k)

        def setSetting(self, k, v):
            return self.setValue(k, v)
    %}
}

%extend rr::SteadyStateSolver {
    %pythoncode %{
        def __dir__(self):
            x = dir(type(self))
            x += self.getSettings()
            return x

        def __getattr__(self, name):
            if(name in self.getSettings()):
                return Solver.getValue(self, name)
            else:
                return _swig_getattr(self, Integrator, name)

        def __setattr__(self, name, value):
            if(name != 'this' and name in self.getSettings()):
                self.setValue(name, value)
            else:
                _swig_setattr(self, Integrator, name, value)

        def __repr__(self):
            return self.toRepr()

        def getSetting(self, k):
            return self.getValue(k)

        def setSetting(self, k, v):
            return self.setValue(k, v)
    %}
}

%extend rr::PyIntegratorListener {
    %pythoncode %{
        __swig_getmethods__["onTimeStep"] = getOnTimeStep
        __swig_setmethods__["onTimeStep"] = setOnTimeStep
        if _newclass: onTimeStep = property(getOnTimeStep, setOnTimeStep)

        __swig_getmethods__["onEvent"] = getOnEvent
        __swig_setmethods__["onEvent"] = setOnEvent
        if _newclass: onEvent = property(getOnEvent, setOnEvent)
     %}
}

%extend rr::PyEventListener {
    %pythoncode %{
        __swig_getmethods__["onTrigger"] = getOnTrigger
        __swig_setmethods__["onTrigger"] = setOnTrigger
        if _newclass: onTrigger = property(getOnTrigger, setOnTrigger)

        __swig_getmethods__["onAssignment"] = getOnAssignment
        __swig_setmethods__["onAssignment"] = setOnAssignment
        if _newclass: onAssignment = property(getOnAssignment, setOnAssignment)
     %}
}


// Copy this code ad verbatim to the Python module
%pythoncode %{
RoadRunner.ensureSolversRegistered()
integrators = list(RoadRunner.getRegisteredIntegratorNames())
steadyStateSolvers = list(RoadRunner.getRegisteredSteadyStateSolverNames())
solvers = integrators + steadyStateSolvers
%}
