/*
 * GPUSimExecutableModel.cpp
 *
 * Author: Andy Somogyi,
 *     email decode: V1 = "."; V2 = "@"; V3 = V1;
 *     andy V1 somogyi V2 gmail V3 com
 */
#pragma hdrstop
#include "GPUSimExecutableModel.h"
#include "ModelResources.h"
#include "GPUSimException.h"
#include "rrSparse.h"
#include "rrLogger.h"
#include "rrException.h"
#include "rrStringUtils.h"
#include "rrConfig.h"
#include <iomanip>
#include <cstdlib>
#include <sstream>

using rr::Logger;
using rr::getLogger;
using rr::LoggingBuffer;
using rr::SelectionRecord;
using rr::EventListener;
using rr::EventListenerPtr;
using rr::EventListenerException;
using rr::Config;

#if defined (_WIN32)
#define isnan _isnan
#else
#define isnan std::isnan
#endif

template <typename numeric_type>
static void dump_array(std::ostream &os, int n, const numeric_type *p)
{
    if (p)
    {
        os << setiosflags(std::ios::floatfield) << std::setprecision(8);
        os << '[';
        for (int i = 0; i < n; ++i)
        {
            os << p[i];
            if (i < n - 1)
            {
                os << ", ";
            }
        }
        os << ']' << std::endl;
    }
    else
    {
        os << "NULL" << std::endl;
    }
}

namespace rr
{

namespace rrgpu
{

  /**
 * checks if the bitfield value has all the flags
 * in type (equiv to checkExact but with a more accurate
 * name)
 */
inline bool checkBitfieldSubset(uint32_t type, uint32_t value) {
    return (value & type) == type;
}

GPUSimExecutableModel::GPUSimExecutableModel(std::string const &sbml, unsigned loadSBMLOptions)
    : GPUSimModel(sbml, loadSBMLOptions)
{
}

GPUSimExecutableModel::~GPUSimExecutableModel()
{
    Log(Logger::LOG_DEBUG) << __FUNC__;
}

string GPUSimExecutableModel::getModelName()
{
    throw_gpusim_exception("not supported");
}

void GPUSimExecutableModel::setTime(double time)
{
    throw_gpusim_exception("not supported");
}

double GPUSimExecutableModel::getTime()
{
    throw_gpusim_exception("not supported");
}


int GPUSimExecutableModel::getNumIndFloatingSpecies()
{
    throw_gpusim_exception("not supported");
}


int GPUSimExecutableModel::getNumDepFloatingSpecies()
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::getNumFloatingSpecies()
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::getNumBoundarySpecies()
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::getNumGlobalParameters()
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::getNumCompartments()
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::getNumReactions()
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::getNumLocalParameters(int reactionId)
{
    throw_gpusim_exception("not supported");
}

void GPUSimExecutableModel::convertToAmounts()
{
    throw_gpusim_exception("not supported");
}

void GPUSimExecutableModel::computeConservedTotals()
{
    throw_gpusim_exception("not supported");
}


int GPUSimExecutableModel::getFloatingSpeciesConcentrations(int len, int const *indx,
        double *values)
{
    throw_gpusim_exception("not supported");
}

void GPUSimExecutableModel::getRateRuleValues(double *rateRuleValues)
{
    throw_gpusim_exception("not supported");
}


void GPUSimExecutableModel::convertToConcentrations()
{
    throw_gpusim_exception("not supported");
}

void GPUSimExecutableModel::updateDependentSpeciesValues()
{
    throw_gpusim_exception("not supported");
}

void GPUSimExecutableModel::computeAllRatesOfChange()
{
    throw_gpusim_exception("not supported");
}

void GPUSimExecutableModel::getStateVectorRate(double time, const double *y, double *dydt)
{
    throw_gpusim_exception("not supported");
}

double GPUSimExecutableModel::getFloatingSpeciesAmountRate(int index,
           const double *reactionRates)
{
    throw_gpusim_exception("not supported");
}

void GPUSimExecutableModel::testConstraints()
{
}

std::string GPUSimExecutableModel::getInfo()
{
    std::stringstream stream;

    double *tmp;

    int nFloat = getNumFloatingSpecies();
    int nBound = getNumBoundarySpecies();
    int nComp = getNumCompartments();
    int nGlobalParam = getNumGlobalParameters();
    int nEvents = getNumEvents();
    int nReactions = getNumReactions();

    stream << "* Calculated Values *" << std::endl;

    tmp = new double[nFloat];
    getFloatingSpeciesAmounts(nFloat, 0, tmp);
    stream << "FloatingSpeciesAmounts:" << std::endl;
    dump_array(stream, nFloat, tmp);

    getFloatingSpeciesConcentrations(nFloat, 0, tmp);
    stream << "FloatingSpeciesConcentrations:" << std::endl;
    dump_array(stream, nFloat, tmp);

    this->getFloatingSpeciesInitConcentrations(nFloat, 0, tmp);
    stream << "FloatingSpeciesInitConcentrations:" << std::endl;
    dump_array(stream, nFloat, tmp);
    delete[] tmp;

    tmp = new double[nReactions];
    getReactionRates(nReactions, 0, tmp);
    stream << "Reaction Rates:" << std::endl;
    dump_array(stream, nReactions, tmp);
    delete[] tmp;

    tmp = new double[nBound];
    getBoundarySpeciesAmounts(nBound, 0, tmp);
    stream << "BoundarySpeciesAmounts:" << std::endl;
    dump_array(stream, nBound, tmp);

    getBoundarySpeciesConcentrations(nBound, 0, tmp);
    stream << "BoundarySpeciesConcentrations:" << std::endl;
    dump_array(stream, nBound, tmp);
    delete[] tmp;

    tmp = new double[nComp];
    getCompartmentVolumes(nComp, 0, tmp);
    stream << "CompartmentVolumes:" << std::endl;
    dump_array(stream, nComp, tmp);

    this->getCompartmentInitVolumes(nComp, 0, tmp);
    stream << "CompartmentInitVolumes:" << std::endl;
    dump_array(stream, nComp, tmp);
    delete[] tmp;

    tmp = new double[nGlobalParam];
    getGlobalParameterValues(nGlobalParam, 0, tmp);
    stream << "GlobalParameters:" << std::endl;
    dump_array(stream, nGlobalParam, tmp);
    delete[] tmp;

    tmp = new double[nGlobalParam];
    getGlobalParameterValues(nGlobalParam, 0, tmp);
    stream << "GlobalParameters:" << std::endl;
    dump_array(stream, nGlobalParam, tmp);
    delete[] tmp;

    unsigned char *tmpEvents = new unsigned char[nEvents];
    getEventTriggers(nEvents, 0, tmpEvents);
    stream << "Events Trigger Status:" << std::endl;
    dump_array(stream, nEvents, (bool*)tmpEvents);
    delete[] tmpEvents;

//     stream << *modelData;

    return stream.str();
}

int GPUSimExecutableModel::getFloatingSpeciesIndex(const string& id)
{
    throw_gpusim_exception("not supported");
}

string GPUSimExecutableModel::getFloatingSpeciesId(int index)
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::getBoundarySpeciesIndex(const string& id)
{
    throw_gpusim_exception("not supported");
}

string GPUSimExecutableModel::getBoundarySpeciesId(int indx)
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::getGlobalParameterIndex(const string& id)
{
    throw_gpusim_exception("not supported");
}

string GPUSimExecutableModel::getGlobalParameterId(int id)
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::getCompartmentIndex(const string& id)
{
    throw_gpusim_exception("not supported");
}

string GPUSimExecutableModel::getCompartmentId(int id)
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::getReactionIndex(const string& id)
{
    throw_gpusim_exception("not supported");
}

string GPUSimExecutableModel::getReactionId(int id)
{
    throw_gpusim_exception("not supported");
}

void GPUSimExecutableModel::evalInitialConditions()
{
    throw_gpusim_exception("not supported");
}

void GPUSimExecutableModel::reset()
{
    throw_gpusim_exception("not supported");
}

void GPUSimExecutableModel::reset(int options)
{

}

bool GPUSimExecutableModel::getConservedSumChanged()
{
    throw_gpusim_exception("not supported");
}

void GPUSimExecutableModel::setConservedSumChanged(bool val)
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::getStateVector(double* stateVector)
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::setStateVector(const double* stateVector)
{
    throw_gpusim_exception("not supported");
}

void GPUSimExecutableModel::print(std::ostream &stream)
{
    stream << "GPUSimExecutableModel" << std::endl;
    stream << getInfo();
}

void GPUSimExecutableModel::getIds(int types, std::list<std::string> &ids)
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::getSupportedIdTypes()
{
    return SelectionRecord::TIME |
        SelectionRecord::BOUNDARY_CONCENTRATION |
        SelectionRecord::FLOATING_CONCENTRATION |
        SelectionRecord::REACTION_RATE |
        SelectionRecord::FLOATING_AMOUNT_RATE |
        SelectionRecord::FLOATING_CONCENTRATION_RATE |
        SelectionRecord::COMPARTMENT |
        SelectionRecord::GLOBAL_PARAMETER |
        SelectionRecord::FLOATING_AMOUNT |
        SelectionRecord::BOUNDARY_AMOUNT |
        SelectionRecord::INITIAL_AMOUNT |
        SelectionRecord::INITIAL_CONCENTRATION |
        SelectionRecord::STOICHIOMETRY;
}

double GPUSimExecutableModel::getValue(const std::string& id)
{
    throw_gpusim_exception("not supported");
}


const rr::SelectionRecord& GPUSimExecutableModel::getSelection(const std::string& str)
{
    throw_gpusim_exception("not supported");
}

void GPUSimExecutableModel::setValue(const std::string& id, double value)
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::getFloatingSpeciesConcentrationRates(int len,
        const int* indx, double* values)
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::setBoundarySpeciesAmounts(int len, const int* indx,
        const double* values)
{
    throw_gpusim_exception("not supported");
}

std::string GPUSimExecutableModel::getStateVectorId(int index)
{
    throw_gpusim_exception("not supported");
}

void GPUSimExecutableModel::evalReactionRates()
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::getNumRateRules()
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::getFloatingSpeciesAmounts(int len, const int* indx,
        double* values)
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::setFloatingSpeciesConcentrations(int len,
        const int* indx, const double* values)
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::getBoundarySpeciesAmounts(int len, const int* indx,
        double* values)
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::getBoundarySpeciesConcentrations(int len,
        const int* indx, double* values)
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::setBoundarySpeciesConcentrations(int len,
        const int* indx, const double* values)
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::getGlobalParameterValues(int len, const int* indx,
        double* values)
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::setGlobalParameterValues(int len, const int* indx,
        const double* values)
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::getCompartmentVolumes(int len, const int* indx,
        double* values)
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::getReactionRates(int len, const int* indx,
        double* values)
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::getNumConservedMoieties()
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::getConservedMoietyIndex(const string& name)
{
    throw_gpusim_exception("not supported");
}

string GPUSimExecutableModel::getConservedMoietyId(int index)
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::getConservedMoietyValues(int len, const int* indx,
        double* values)
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::setConservedMoietyValues(int len, const int* indx,
        const double* values)
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::getFloatingSpeciesAmountRates(int len,
        int const *indx, double *values)
{
    throw_gpusim_exception("not supported");
}


int GPUSimExecutableModel::setFloatingSpeciesAmounts(int len, int const *indx,
        const double *values)
{
    throw_gpusim_exception("not supported");
}


int GPUSimExecutableModel::setCompartmentVolumes(int len, const int* indx,
        const double* values)
{
    throw_gpusim_exception("not supported");
}



double GPUSimExecutableModel::getStoichiometry(int speciesIndex, int reactionIndex)
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::getStoichiometryMatrix(int* pRows, int* pCols,
        double** pData)
{
    throw_gpusim_exception("not supported");
}



/******************************* Initial Conditions Section *******************/
#if (1) /**********************************************************************/
/******************************************************************************/


int GPUSimExecutableModel::setFloatingSpeciesInitConcentrations(int len,
        const int* indx, const double* values)
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::getFloatingSpeciesInitConcentrations(int len,
        const int* indx, double* values)
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::setFloatingSpeciesInitAmounts(int len, int const *indx,
            double const *values)
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::getFloatingSpeciesInitAmounts(int len, int const *indx,
                double *values)
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::setCompartmentInitVolumes(int len, const int *indx,
            double const *values)
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::getCompartmentInitVolumes(int len, const int *indx,
                double *values)
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::setGlobalParameterInitValues(int len, const int* indx,
        const double* values)
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::getGlobalParameterInitValues(int len, const int *indx,
                double *values)
{
    throw_gpusim_exception("not supported");
}

/******************************* End Initial Conditions Section ***************/
#endif /***********************************************************************/
/******************************************************************************/

/******************************* Events Section *******************************/
#if (1) /**********************************************************************/
/******************************************************************************/

int GPUSimExecutableModel::getNumEvents()
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::getEventTriggers(int len, const int *indx, unsigned char *values)
{
    throw_gpusim_exception("not supported");
}

void GPUSimExecutableModel::applyEvents(double timeEnd, const unsigned char* previousEventStatus,
	    const double *initialState, double* finalState)
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::applyPendingEvents(const double *stateVector, double timeEnd, double tout)
{
    throw_gpusim_exception("not supported");
}

void GPUSimExecutableModel::getEventRoots(double time, const double* y, double* gdot)
{
    throw_gpusim_exception("not supported");
}

double GPUSimExecutableModel::getNextPendingEventTime(bool pop)
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::getPendingEventSize()
{
    throw_gpusim_exception("not supported");
}

void GPUSimExecutableModel::resetEvents()
{
    throw_gpusim_exception("not supported");
}

int GPUSimExecutableModel::getEventIndex(const std::string& eid)
{
    throw_gpusim_exception("not supported");
}

std::string GPUSimExecutableModel::getEventId(int index)
{
    throw_gpusim_exception("not supported");
}

void GPUSimExecutableModel::setEventListener(int index, rr::EventListenerPtr eventHandler)
{
    throw_gpusim_exception("not supported");
}

rr::EventListenerPtr GPUSimExecutableModel::getEventListener(int index)
{
    throw_gpusim_exception("not supported");
}

/******************************* Events Section *******************************/
  #endif /**********************************************************************/
/******************************************************************************/

} // namespace rrgpu

} // namespace rr