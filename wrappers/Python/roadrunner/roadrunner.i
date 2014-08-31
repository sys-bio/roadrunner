
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
    #include <numpy/arrayobject.h>
    #include <lsMatrix.h>
    #include <lsLibla.h>
    #include <lsLA.h>
    #include <lsLUResult.h>
    #include <lsUtils.h>
    #include <rrCompiler.h>
    #include <ModelGenerator.h>
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
    #include "PyUtils.h"
    #include "PyLoggerStream.h"

    // make a python obj out of the C++ ExecutableModel, this is used by the PyEventListener
    // class. This function is defined later in this compilation unit.
    PyObject *ExecutableModel_NewPythonObj(rr::ExecutableModel*);
    PyObject *Integrator_NewPythonObj(rr::Integrator*);


    #include "PyEventListener.h"
    #include "PyIntegratorListener.h"

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

 // the integrator listener is a shared ptr
#define SWIG_SHARED_PTR_SUBNAMESPACE tr1
%include "std_shared_ptr.i"

%shared_ptr(rr::PyIntegratorListener)



%template(IntVector) std::vector<int>;
%template(StringVector) std::vector<std::string>;
%template(StringList) std::list<std::string>;

%apply std::vector<std::string> {vector<std::string>, vector<string>, std::vector<string> };

//%template(SelectionRecordVector) std::vector<rr::SelectionRecord>;
//%apply std::vector<rr::SelectionRecord> {std::vector<SelectionRecord>, std::vector<rr::SelectionRecord>, vector<SelectionRecord>};

%apply std::list<std::string>& OUTPUT {std::list<std::string>};

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

    int rows = ($1).numRows();
    int cols = ($1).numCols();
    int nd = 2;
    npy_intp dims[2] = {rows, cols};
    double *data = (double*)malloc(sizeof(double)*rows*cols);
    memcpy(data, ($1).getArray(), sizeof(double)*rows*cols);

    PyObject *pArray = PyArray_New(&PyArray_Type, nd, dims, NPY_DOUBLE, NULL, data, 0,
            NPY_CARRAY | NPY_OWNDATA, NULL);
    VERIFY_PYARRAY(pArray);
    $result  = pArray;
}


/**
 * Convert from C --> Python
 * reference roadrunner owned data.
 */
%typemap(out) const ls::DoubleMatrix* {

    int rows = ($1)->numRows();
    int cols = ($1)->numCols();
    int nd = 2;
    npy_intp dims[2] = {rows, cols};
    double *data = ($1)->getArray();

    PyObject *pArray = PyArray_New(&PyArray_Type, nd, dims, NPY_DOUBLE, NULL, data, 0,
            NPY_CARRAY, NULL);
    VERIFY_PYARRAY(pArray);
    $result  = pArray;
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



%typemap(in) const rr::Variant& (rr::Variant temp) {

    try {
        temp = Variant_from_py($input);
        $1 = &temp;
    } catch (const std::exception& e) {
        SWIG_exception(SWIG_RuntimeError, e.what());
    }
}

%apply const rr::Variant& {rr::Variant&, Variant&, const Variant&};





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
//%ignore rr::RoadRunner::getEigenvalueIds;
%ignore rr::RoadRunner::getNumberOfFloatingSpecies;
//%ignore rr::RoadRunner::getUnscaledElasticityMatrix;
%ignore rr::RoadRunner::setTimeStart;
//%ignore rr::RoadRunner::getEigenvalues;
%ignore rr::RoadRunner::getNumberOfGlobalParameters;
//%ignore rr::RoadRunner::getUnscaledFluxControlCoefficientMatrix;
//%ignore rr::RoadRunner::setValue;
%ignore rr::RoadRunner::getEigenvaluesCpx;
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
%ignore rr::RoadRunner::getFloatingSpeciesConcentrations;
%ignore rr::RoadRunner::getRateOfChange;
//%ignore rr::RoadRunner::getlibSBMLVersion;
//%ignore rr::RoadRunner::writeSBML;
%ignore rr::RoadRunner::getSimulateOptions;
%ignore rr::RoadRunner::setSimulateOptions;
%ignore rr::RoadRunner::getIds(int types, std::list<std::string> &);

%ignore rr::RoadRunner::getOptions;

%ignore rr::RoadRunner::simulate;

%rename (_getCurrentIntegrator) rr::RoadRunner::getIntegrator();
%rename (_getIntegrator) rr::RoadRunner::getIntegrator(SimulateOptions::Integrator);
%rename (_load) rr::RoadRunner::load;


%ignore rr::Config::getInt;
%ignore rr::Config::getString;
%ignore rr::Config::getBool;
%ignore rr::Config::getDouble;

// ignore SimulateOptions key access methods,
// these are replaced by python dictionary protocol.
%ignore rr::SimulateOptions::setValue;
%ignore rr::SimulateOptions::getValue;
%ignore rr::SimulateOptions::hasKey;
%ignore rr::SimulateOptions::deleteValue;
%ignore rr::SimulateOptions::getKeys;

%rename (_setIntegratorId) rr::SimulateOptions::setIntegrator;

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

%rename rr::conversion::ConservedMoietyConverter PyConservedMoietyConverter;

%ignore rr::ostream;
%ignore ostream;
%ignore std::ostream;
%ignore operator<<(ostream&, const rr::SelectionRecord& rec);
%ignore operator<<(rr::ostream&, const rr::SelectionRecord& rec);

// ignore all instances of the Configurable methods.
%ignore *::createConfigNode;
%ignore *::loadConfig;

// Warning 389: operator[] ignored (consider using %extend)
// Warning 401: Nothing known about base class 'Configurable'. Ignored.


// Warning 315: Nothing known about 'std::ostream'.
namespace std { class ostream{}; }


// Warning 401: Nothing known about base class 'Configurable'. Ignored.


/**
 * include the roadrunner files here, this is where the wrappers are generated.
 */

//%import(module="roadrunner") "Configurable.h"
//%ignore rr::Configurable;

//namespace rr { class Configurable{}; }

%ignore rr::Configurable;
%include <Configurable.h>

%include <rrRoadRunnerOptions.h>
%include <rrLogger.h>
%include <rrCompiler.h>
%include <rrExecutableModel.h>
%include <ModelGenerator.h>
%include <rrVersionInfo.h>

%thread;
%include <rrRoadRunner.h>
%nothread;

%include <rrSelectionRecord.h>
%include <conservation/ConservedMoietyConverter.h>
%include <Integrator.h>

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

    PyObject* _simulate(const rr::SimulateOptions* opt) {
        // its not const correct...
        ls::DoubleMatrix *result = const_cast<ls::DoubleMatrix*>($self->simulate(opt));

        // a valid array descriptor:
        // In [87]: b = array(array([0,1,2,3]),
        //      dtype=[('r', 'f8'), ('g', 'f8'), ('b', 'f8'), ('a', 'f8')])


        // are we returning a structured array?
        if (opt->flags & SimulateOptions::STRUCTURED_RESULT) {

            // get the column names
            const std::vector<SelectionRecord>& sel = ($self)->getSelections();
            std::vector<string> names(sel.size());

            for(int i = 0; i < sel.size(); ++i) {
                names[i] = sel[i].to_string();
            }


            int rows = result->numRows();
            int cols = result->numCols();

            if (cols == 0) {
                Py_RETURN_NONE;
            }

            double* mData = result->getArray();

            PyObject* list = PyList_New(names.size());

            for(int i = 0; i < names.size(); ++i)
            {
                PyObject *col = PyString_FromString(names[i].c_str());
                PyObject *type = PyString_FromString("f8");
                PyObject *tup = PyTuple_Pack(2, col, type);

                Py_DECREF(col);
                Py_DECREF(type);

                // list takes ownershipt of tuple
                void PyList_SET_ITEM(list, i, tup);
            }

            PyArray_Descr* descr = 0;
            PyArray_DescrConverter(list, &descr);

            // done with list
            Py_CLEAR(list);
            npy_intp dims[] = {rows};

            // steals a reference to descr
            PyObject *pyres = PyArray_SimpleNewFromDescr(1, dims,  descr);
            VERIFY_PYARRAY(pyres);

            if (pyres) {

                assert(PyArray_NBYTES(pyres) == rows*cols*sizeof(double) && "invalid array size");

                double* data = (double*)PyArray_BYTES(pyres);

                memcpy(data, mData, rows*cols*sizeof(double));
            }

            return pyres;
        }
        // standard array result.
        // this version just wraps the roadrunner owned data.
        else {

            int rows = result->numRows();
            int cols = result->numCols();
            int nd = 2;
            npy_intp dims[2] = {rows, cols};
            PyObject *pArray = NULL;

            if (opt->flags & SimulateOptions::COPY_RESULT) {

                Log(rr::Logger::LOG_DEBUG) << "copying result data";

                pArray = PyArray_SimpleNew(2, dims, NPY_DOUBLE);

                VERIFY_PYARRAY(pArray);
                assert(PyArray_NBYTES(pArray) == rows*cols*sizeof(double) && "invalid array size");

                double *pyData = (double*)PyArray_BYTES(pArray);
                double *mData = result->getArray();

                memcpy(pyData, mData, rows*cols*sizeof(double));
            }
            else {

                Log(rr::Logger::LOG_DEBUG) << "wraping existing data";

                double *data = result->getArray();

                pArray = PyArray_New(&PyArray_Type, nd, dims, NPY_DOUBLE, NULL, data, 0,
                                     NPY_CARRAY, NULL);
                VERIFY_PYARRAY(pArray);

            }
            return pArray;
        }
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

        __swig_getmethods__["selections"] = _getSelections
        __swig_setmethods__["selections"] = _setSelections
        __swig_getmethods__["steadyStateSelections"] = _getSteadyStateSelections
        __swig_setmethods__["steadyStateSelections"] = _setSteadyStateSelections
        __swig_getmethods__["conservedMoietyAnalysis"] = _getConservedMoietyAnalysis
        __swig_setmethods__["conservedMoietyAnalysis"] = _setConservedMoietyAnalysis
        __swig_getmethods__["model"] = _getModel
        __swig_getmethods__["integrator"] = _getCurrentIntegrator

        if _newclass:
            selections = property(_getSelections, _setSelections)
            steadyStateSelections = property(_getSteadyStateSelections, _setSteadyStateSelections)
            conservedMoietyAnalysis=property(_getConservedMoietyAnalysis, _setConservedMoietyAnalysis)
            model = property(getModel)
            integrator = property(_getCurrentIntegrator)


        # static list of properties added to the RoadRunner
        # class object
        _properties = []

        def _makeProperties(self):

            #global _properties
       
            # always clear the old properties
            for s in RoadRunner._properties:
                del RoadRunner.__swig_getmethods__[s]
                del RoadRunner.__swig_setmethods__[s]
                delattr(RoadRunner, s)

            # properties now empty
            RoadRunner._properties = []

            # check if we should make new properties
            if Config.getValue(Config.ROADRUNNER_DISABLE_PYTHON_DYNAMIC_PROPERTIES):
                return

            model = self.getModel()

            # can't make properties without a model. 
            if model is None:
                return 

            def mk_fget(sel): return lambda self: model.__getitem__(sel)
            def mk_fset(sel): return lambda self, val: model.__setitem__(sel, val)

            def makeProperty(name, sel):
                fget = mk_fget(sel)
                fset = mk_fset(sel)
                RoadRunner.__swig_getmethods__[name] = fget
                RoadRunner.__swig_setmethods__[name] = fset
                setattr(RoadRunner, name, property(fget, fset))
                RoadRunner._properties.append(name)

            for s in model.getFloatingSpeciesIds():
                makeProperty(s, "[" + s + "]")  # concentrations
                makeProperty(s + "_amt", s)     # amounts


            for s in model.getBoundarySpeciesIds():
                makeProperty(s, "[" + s + "]")  # concentrations
                makeProperty(s + "_amt", s)     # amounts


            for s in model.getGlobalParameterIds() + model.getCompartmentIds() + model.getReactionIds():
                makeProperty(s, s)



        # Set up the python dyanic properties for model access, 
        # save the original init method
        _swig_init = __init__  

        def _new_init(self, *args):
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

        def getIntegrator(self, iname=None):
            """
            Get the integrator based on its name.
            """
            if iname is None:
                return self._getCurrentIntegrator()

            id = SimulateOptions.getIntegratorIdFromName(iname)
            return self._getIntegrator(id)

        def setIntegrator(self, iname):
            """
            set the default integrator.
            """
            self.simulateOptions.integrator = iname

            if self.model is None:
                Logger.log(Logger.LOG_WARNING, "Setting integrator without a model, changes will take effect when a model is loaded")

        def simulate(self, *args, **kwargs):
            """
            Simulate the optionally plot current SBML model. This is the one stop shopping method
            for simulation and ploting.

            simulate accepts a up to four positional arguments and a large number of keyword args.

            The first four (optional) arguments are treated as:

            1: Start Time, if this is a number.

            2: End Time, if this is a number.

            3: Number of Steps, if this is a number.

            4: List of Selections.

            All four of the positional arguments are optional. If any of the positional arguments are
            a list of string instead of a number, then they are interpreted as a list of selections.


            There are a number of ways to call simulate.

            1. With no arguments. In this case, the current set of `SimulateOptions` will
            be used for the simulation. The current set may be changed either directly
            via setSimulateOptions() or with one of the two alternate ways of calling
            simulate.

            2: With single `SimulateOptions` argument. In this case, all of the settings
            in the given options are copied and will be used for the current and future
            simulations.

            3: With the three positions arguments, `timeStart`, `timeEnd`, `steps`. In this case
            these three values are copied and will be used for the current and future simulations.

            4: With keyword arguments where keywords are the property names of the SimulateOptions
            class. To reset the model, simulate from 0 to 10 in 1000 steps and plot we can::

                rr.simulate(end=10, start=0, steps=1000, resetModel=True, plot=True)

            The options given in the 2nd and 3rd forms will remain in effect until changed. So, if
            one calls::

                rr.simulate (0, 3, 100)

            The start time of 0, end time of 3 and steps of 100 will remain in effect, so that if this
            is followed by a call to::

                rr.simulate()

            This simulation will use the previous values.

            simulate accepts the following list of keyword arguments:

            integrator
                A text string specifying which integrator to use. Currently supports "cvode"
                for deterministic simulation (default) and "gillespie" for stochastic
                simulation.

            sel or selections
                A list of strings specifying what values to display in the output.

            plot
                True or False
                If True, RoadRunner will create a basic plot of the simulation result using
                the built in plot routine which uses MatPlotLib.

            absolute
                A number representing the absolute difference permitted for the integrator
                tolerance.

            duration
                The duration of the simulation run, in the model's units of time.
                Note, setting the duration automatically sets the end time and visa versa.

            end
                The simulation end time. Note, setting the end time automatically sets
                the duration accordingly and visa versa.

            relative
                A float-point number representing the relative difference permitted.
                Defaults 0.0001

            resetModel (or just "reset"???)
                True or False
                Causes the model to be reset to the original conditions specified in
                the SBML when the simulation is run.

            start
                The start time of the simulation time-series data. Often this is 0,
                but not necessarily.

            steps
                The number of steps at which the output is sampled. The samples are evenly spaced.
                When a simulation system calculates the data points to record, it will typically
                divide the duration by the number of time steps. Thus, for N steps, the output
                will have N+1 data rows.

            stiff
                True or False
                Use the stiff integrator. Only use this if the model is stiff and causes issues
                with the regular integrator. The stiff integrator is slower than the conventional
                integrator.

            multiStep
                True or False
                Perform a multi step integration.
                * Experimental *
                Perform a multi-step simulation. In multi-step simulation, one may monitor the
                variable time stepping via the IntegratorListener events system.

            initialTimeStep
                A user specified initial time step. If this is <= 0, the integrator will attempt
                to determine a safe initial time step.

                Note, for each number of steps given to RoadRunner.simulate or RoadRunner.integrate
                the internal integrator may take many many steps to reach one of the external time steps.
                This value specifies an initial value for the internal integrator time step.

            minimumTimeStep
                Specify the minimum time step that the internal integrator will use.
                Uses integrator estimated value if <= 0.

            maximumTimeStep
                Specify the maximum time step size that the internal integrator will use.
                Uses integrator estimated value if <= 0.

            maximumNumSteps
                Specify the maximum number of steps the internal integrator will use before
                reaching the user specified time span. Uses the integrator default value if <= 0.

            seed
                Specify a seed to use for the random number generator for stochastic simulations.
                The seed is used whenever the integrator is reset, i.e. `r.reset()`.
                If no seed is specified, the current system time is used for seed.


            :returns: a numpy array with each selected output time series being a
             column vector, and the 0'th column is the simulation time.
            :rtype: numpy.ndarray
            """

            doPlot = False
            show = True

            # user specified number of steps via 3rd arg or steps=xxx
            haveSteps = False

            # variableStep = True was specified in args
            haveVariableStep = False
            o = self.simulateOptions

            # did the options originally have a seed, if so, don't delete it when we're done
            hadSeed = "seed" in o

            # check if we have just a sim options
            if len(args) >= 1:
                if type(args[0]) == type(self.simulateOptions):
                    o = args[0]
                elif type(args[0]) == list:
                    # its a selection list
                    self.selections = args[0]
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
                    self.selections = args[1]
                elif isinstance(args[1], (int, float)):
                    # treat it as a number
                    o.end = args[1]
                else:
                    raise ValueError("argument 2 must be either a number, list or "
                                     "SimulateOptions object, recieved: {0}".format(str(args[1])))


            # third arg is treated as number of steps
            if len(args) >= 3:
                if type(args[2]) == list:
                    # its a selection list
                    self.selections = args[2]
                elif isinstance(args[2], (int, float)):
                    # treat it as a number
                    o.steps = args[2]
                    haveSteps = True
                else:
                    raise ValueError("argument 3 must be either a number, list or "
                                     "SimulateOptions object, recieved: {0}".format(str(args[2])))

            # forth arg may be a list (currently)
            if len(args) >= 4:
                if type(args[3]) == list:
                    # its a selection list
                    self.selections = args[3]
                else:
                    raise ValueError("argument 4 (if given) must be a list of selections "
                                     ", recieved: {0}".format(str(args[3])))


            # go through the list of keyword args
            for k,v in kwargs.iteritems():

                # changing integrators.
                if k == "integrator":
                    if type(v) == str:
                        # this automatically sets the variable / fixed time step
                        # according to integrator type, raises exception if invalid
                        # integrator string.
                        o.integrator = v
                    else:
                        raise Exception("{0} is invalid argument for integrator, integrator name must be a string.".format(v))
                    continue

                # specifying selections:
                if k == "selections" or k == "sel":
                    self.selections = v
                    continue

                # reset model, also accept 'reset'
                if k == "reset" or k == "resetModel":
                    o.resetModel = v
                    continue

                # check if variableStep was explicitly specified, this overrides the steps
                # positional arg
                if k == "variableStep":
                    haveVariableStep = True
                    o.variableStep = v
                    continue

                # check if specifying seed for RNG.
                if k == "seed":
                    o["seed"] = v
                    continue

                # look through all the attributes of the SimulateOptions class,
                # if its one of these, set it.
                if SimulateOptions.__dict__.has_key(k):
                    setattr(o, k, v)
                    continue

                if k == "plot":
                    doPlot = v
                    continue

                if k == "show":
                    show = v
                    continue

                raise Exception("{0} is not a valid keyword argument".format(k))


            # if we are doing a stochastic sim,
            # explicit options of variableStep trumps everything,
            # if not explicit, variableStep is if number of steps was specified,
            # if no steps, varStep = true, false otherwise.
            if SimulateOptions.getIntegratorType(o.getIntegratorId()) == \
                SimulateOptions.STOCHASTIC and not haveVariableStep:
                o.variableStep = not haveSteps

            # the options are set up, now actually run the simuation...
            result = self._simulate(o)

            if not hadSeed:
                del o["seed"]

            if doPlot:
                self.plot(show)

            return result

        def getAvailableIntegrators(self):
            """
            get a list of available integrator names.
            """
            return [SimulateOptions.getIntegratorNameFromId(i) \
                for i in range(0, SimulateOptions.INTEGRATOR_END)]


        def plot(self, show=True):
            """
            RoadRunner.plot([show])

            Plot the previously run simulation result using Matplotlib.

            This takes the contents of the simulation result and builds a
            legend from the selection list.


            If the optional prameter 'show' [default is True] is given, the pylab
            show() method is called.
            """

            import pylab as p

            result = self.getSimulationData()

            if result is None:
                raise Exception("no simulation result")

            # check if standard numpy array
            if result.dtype.names is None:

                selections = self.selections

                if len(result.shape) != 2 or result.shape[1] != len(selections):
                    raise Exception("simulation result columns not equal to number of selections, likely a simulation has not been run")

                times = result[:,0]

                for i in range(1, len(selections)):
                    series = result[:,i]
                    name = selections[i]
                    p.plot(times, series, label=str(name))

            # result is structured array
            else:
                if len(result.dtype.names) < 1:
                    raise Exception('no columns to plot')

                time = result.dtype.names[0]

                for name in result.dtype.names[1:]:
                    p.plot(result[time], result[name], label=name)

            p.legend()

            if show:
                p.show()
    %}
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

    PyObject *keys() {
        std::vector<std::string> keys = $self->getKeys();

        unsigned size = keys.size();

        PyObject* pyList = PyList_New(size);

        unsigned j = 0;

        for (std::vector<std::string>::const_iterator i = keys.begin(); i != keys.end(); ++i)
        {
            const std::string& key  = *i;
            PyObject* pyStr = PyString_FromString(key.c_str());
            PyList_SET_ITEM(pyList, j++, pyStr);
        }

        return pyList;
    }

    const rr::Variant& __getitem__(const std::string& id) {
        return ($self)->getValue(id);
    }

    void __setitem__(const std::string& key, const rr::Variant& value) {
        ($self)->setValue(key, value);
    }

    void __delitem__(const std::string& key) {
        ($self)->deleteValue(key);
    }

    bool __contains__(const std::string& key) {
        return $self->hasKey(key);
    }

    std::string _getIntegrator() {
        return SimulateOptions::getIntegratorNameFromId(($self)->integrator);
    }

    void _setIntegrator(const std::string &str) {

        // set the value
        SimulateOptions::Integrator value = SimulateOptions::getIntegratorIdFromName(str);

        ($self)->setIntegrator(value);
    }

    rr::SimulateOptions::Integrator getIntegratorId() {
        return ($self)->integrator;
    }


    %pythoncode %{
        def getListener(self):
            return self._getListener()

        def setListener(self, listener):
            if listener is None:
                self._clearListener()
            else:
                self._setListener(listener)

        __swig_getmethods__["integrator"] = _getIntegrator
        __swig_setmethods__["integrator"] = _setIntegrator
        if _newclass:
            integrator = property(_getIntegrator, _setIntegrator)
    %}



}

%{
    double rr_SimulateOptions_end_get(SimulateOptions* opt) {
        return opt->start + opt->duration;
    }

    void rr_SimulateOptions_end_set(SimulateOptions* opt, double end) {
        opt->duration = end - opt->start;
    }

    bool rr_SimulateOptions_resetModel_get(SimulateOptions* opt) {
        return opt->flags & SimulateOptions::RESET_MODEL;
    }

    void rr_SimulateOptions_resetModel_set(SimulateOptions* opt, bool value) {
        if (value) {
            opt->flags |= SimulateOptions::RESET_MODEL;
        } else {
            opt->flags &= !SimulateOptions::RESET_MODEL;
        }
    }

    bool rr_SimulateOptions_structuredResult_get(SimulateOptions* opt) {
        return opt->flags & SimulateOptions::STRUCTURED_RESULT;
    }

    void rr_SimulateOptions_structuredResult_set(SimulateOptions* opt, bool value) {
        if (value) {
            opt->flags |= SimulateOptions::STRUCTURED_RESULT;
        } else {
            opt->flags &= !SimulateOptions::STRUCTURED_RESULT;
        }
    }

    bool rr_SimulateOptions_stiff_get(SimulateOptions* opt) {
        return opt->integratorFlags & SimulateOptions::STIFF;
    }

    void rr_SimulateOptions_stiff_set(SimulateOptions* opt, bool value) {
        if (value) {
            opt->integratorFlags |= SimulateOptions::STIFF;
        } else {
            opt->integratorFlags &= ~SimulateOptions::STIFF;
        }
    }

    bool rr_SimulateOptions_multiStep_get(SimulateOptions* opt) {
        return opt->integratorFlags & SimulateOptions::MULTI_STEP;
    }

    void rr_SimulateOptions_multiStep_set(SimulateOptions* opt, bool value) {
        if (value) {
            opt->integratorFlags |= SimulateOptions::MULTI_STEP;
        } else {
            opt->integratorFlags &= ~SimulateOptions::MULTI_STEP;
        }
    }

    bool rr_SimulateOptions_copyResult_get(SimulateOptions* opt) {
        return opt->flags & SimulateOptions::COPY_RESULT;
    }

    void rr_SimulateOptions_copyResult_set(SimulateOptions* opt, bool value) {
        if (value) {
            opt->flags |= SimulateOptions::COPY_RESULT;
        } else {
            opt->flags &= ~SimulateOptions::COPY_RESULT;
        }
    }


    bool rr_SimulateOptions_variableStep_get(SimulateOptions* opt) {
        return opt->integratorFlags & SimulateOptions::VARIABLE_STEP;
    }

    void rr_SimulateOptions_variableStep_set(SimulateOptions* opt, bool value) {
        if (value) {
            opt->integratorFlags |= SimulateOptions::VARIABLE_STEP;
        } else {
            opt->integratorFlags &= ~SimulateOptions::VARIABLE_STEP;
        }
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
        return rr_ExecutableModel_getIds($self, rr::SelectionRecord::CONSREVED_MOIETY);
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

    PyObject *keys() {
        std::vector<std::string> keys = $self->getKeys();

        unsigned size = keys.size();

        PyObject* pyList = PyList_New(size);

        unsigned j = 0;

        for (std::vector<std::string>::const_iterator i = keys.begin(); i != keys.end(); ++i)
        {
            const std::string& key  = *i;
            PyObject* pyStr = PyString_FromString(key.c_str());
            PyList_SET_ITEM(pyList, j++, pyStr);
        }

        return pyList;
    }

    const rr::Variant __getitem__(const std::string& id) {
        return ($self)->getValue(id);
    }

    void __setitem__(const std::string& key, const rr::Variant& value) {
        ($self)->setValue(key, value);
    }

    void __delitem__(const std::string& key) {
        ($self)->deleteValue(key);
    }

    bool __contains__(const std::string& key) {
        return $self->hasKey(key);
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
        __swig_getmethods__["name"] = getName
        if _newclass:
            listener = property(getListener, setListener)
            name = property(getName)
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




