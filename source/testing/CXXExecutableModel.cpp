/*
 * CXXExecutableModel.cpp
 *
 *  Created on: Dec 11, 2014
 *      Author: andy
 */

#include "CXXExecutableModel.h"
#include "rrLogger.h"
#include "rrOSSpecifics.h"

using rr::Logger;

namespace rrtesting
{

CXXExecutableModel::CXXExecutableModel(const rr::Dictionary* dict)
{
    Log(Logger::LOG_NOTICE) << __FUNC__;
}

CXXExecutableModel::~CXXExecutableModel()
{
    Log(Logger::LOG_NOTICE) << __FUNC__;
}

std::string CXXExecutableModel::getModelName()
{
    return "C++ Test Model";
}

void CXXExecutableModel::setTime(double _time)
{
}

double CXXExecutableModel::getTime()
{
    return 0;
}

void CXXExecutableModel::reset()
{
}

int CXXExecutableModel::getNumDepFloatingSpecies()
{
    return 0;
}

int CXXExecutableModel::getNumFloatingSpecies()
{
    return 0;
}

int CXXExecutableModel::getFloatingSpeciesIndex(const std::string& eid)
{
    return 0;
}

std::string CXXExecutableModel::getFloatingSpeciesId(int index)
{
    return "";
}

int CXXExecutableModel::getNumIndFloatingSpecies()
{
    return 0;
}

int CXXExecutableModel::getFloatingSpeciesAmounts(size_t len, const int* indx,
        double* values)
{
    return 0;
}

int CXXExecutableModel::setFloatingSpeciesAmounts(size_t len, const int* indx,
        const double* values)
{
    return 0;
}

int CXXExecutableModel::getFloatingSpeciesAmountRates(size_t len, const int* indx,
        double* values)
{
    return 0;
}

int CXXExecutableModel::getFloatingSpeciesConcentrationRates(size_t len,
        const int* indx, double* values)
{
    return 0;
}

int CXXExecutableModel::getFloatingSpeciesConcentrations(size_t len,
        const int* indx, double* values)
{
    return 0;
}

int CXXExecutableModel::setFloatingSpeciesConcentrations(size_t len,
        const int* indx, const double* values)
{
    return 0;
}

int CXXExecutableModel::setFloatingSpeciesInitConcentrations(size_t len,
        const int* indx, const double* values)
{
    return 0;
}

int CXXExecutableModel::getFloatingSpeciesInitConcentrations(size_t len,
        const int* indx, double* values)
{
    return 0;
}

int CXXExecutableModel::setFloatingSpeciesInitAmounts(size_t len, const int* indx,
        const double* values)
{
    return 0;
}

int CXXExecutableModel::getFloatingSpeciesInitAmounts(size_t len, const int* indx,
        double* values)
{
    return 0;
}

int CXXExecutableModel::getNumBoundarySpecies()
{
    return 0;
}

int CXXExecutableModel::getBoundarySpeciesIndex(const std::string& eid)
{
    return 0;
}

std::string CXXExecutableModel::getBoundarySpeciesId(int index)
{
    return "";
}

int CXXExecutableModel::getBoundarySpeciesAmounts(size_t len, const int* indx,
        double* values)
{
    return 0;
}

int CXXExecutableModel::getBoundarySpeciesConcentrations(size_t len,
        const int* indx, double* values)
{
    return 0;
}

int CXXExecutableModel::setBoundarySpeciesConcentrations(size_t len,
        const int* indx, const double* values)
{
    return 0;
}

int CXXExecutableModel::getNumGlobalParameters()
{
    return 0;
}

int CXXExecutableModel::getGlobalParameterIndex(const std::string& eid)
{
    return 0;
}

std::string CXXExecutableModel::getGlobalParameterId(int index)
{
    return "";
}

int CXXExecutableModel::getGlobalParameterValues(size_t len, const int* indx,
        double* values)
{
    return 0;
}

int CXXExecutableModel::setGlobalParameterValues(size_t len, const int* indx,
        const double* values)
{
    return 0;
}

int CXXExecutableModel::getNumCompartments()
{
    return 0;
}

int CXXExecutableModel::getCompartmentIndex(const std::string& eid)
{
    return 0;
}

std::string CXXExecutableModel::getCompartmentId(int index)
{
    return "";
}

int CXXExecutableModel::getCompartmentVolumes(size_t len, const int* indx,
        double* values)
{
    return 0;
}

int CXXExecutableModel::setCompartmentVolumes(size_t len, const int* indx,
        const double* values)
{
    return 0;
}

int CXXExecutableModel::setCompartmentInitVolumes(size_t len, const int* indx,
        const double* values)
{
    return 0;
}

int CXXExecutableModel::getCompartmentInitVolumes(size_t len, const int* indx,
        double* values)
{
    return 0;
}

void CXXExecutableModel::getIds(int types, std::list<std::string>& ids)
{
}

int CXXExecutableModel::getSupportedIdTypes()
{
    return 0;
}

double CXXExecutableModel::getValue(const std::string& id)
{
    return 0;
}

void CXXExecutableModel::setValue(const std::string& id, double value)
{
}

int CXXExecutableModel::getStoichiometryMatrix(int* rows, int* cols,
        double** data)
{
    return 0;
}

double CXXExecutableModel::getStoichiometry(int speciesIndex, int reactionIndex)
{
    return 0;
}

int CXXExecutableModel::getNumConservedMoieties()
{
    return 0;
}

int CXXExecutableModel::getConservedMoietyIndex(const std::string& eid)
{
    return 0;
}

std::string CXXExecutableModel::getConservedMoietyId(int index)
{
    return "";
}

int CXXExecutableModel::getConservedMoietyValues(size_t len, const int* indx,
        double* values)
{
    return 0;
}

int CXXExecutableModel::setConservedMoietyValues(size_t len, const int* indx,
        const double* values)
{
    return 0;
}

int CXXExecutableModel::getNumRateRules()
{
    return 0;
}

int CXXExecutableModel::getNumReactions()
{
    return 0;
}

int CXXExecutableModel::getReactionIndex(const std::string& eid)
{
    return 0;
}

std::string CXXExecutableModel::getReactionId(size_t index)
{
    return "";
}

int CXXExecutableModel::getReactionRates(size_t len, const int* indx,
        double* values)
{
    return 0;
}

void CXXExecutableModel::getRateRuleValues(double* rateRuleValues)
{
}

std::string CXXExecutableModel::getStateVectorId(int index)
{
    return "";
}

int CXXExecutableModel::getStateVector(double* stateVector)
{
    return 0;
}

int CXXExecutableModel::setStateVector(const double* stateVector)
{
    return 0;
}

void CXXExecutableModel::getStateVectorRate(double time, const double* y,
        double* dydt)
{
}

void CXXExecutableModel::testConstraints()
{
}

std::string CXXExecutableModel::getInfo()
{
    return "C++ Test Model";
}

void CXXExecutableModel::print(std::ostream& stream)
{
}

int CXXExecutableModel::getNumEvents()
{
    return 0;
}

int CXXExecutableModel::getEventTriggers(size_t len, const int* indx,
        unsigned char* values)
{
    return 0;
}

int CXXExecutableModel::applyEvents(double timeEnd,
        const unsigned char* previousEventStatus, const double* initialState,
        double* finalState)
{
    return 0;
}

int CXXExecutableModel::applyPendingEvents(double timeEnd)
{
    return 0;
}

void CXXExecutableModel::getEventRoots(double time, const double* y,
        double* gdot)
{
}

double CXXExecutableModel::getNextPendingEventTime(bool pop)
{
    return 0;
}

int CXXExecutableModel::getPendingEventSize()
{
    return 0;
}

void CXXExecutableModel::resetEvents()
{
}

int CXXExecutableModel::getEventIndex(const std::string& eid)
{
    return 0;
}

std::string CXXExecutableModel::getEventId(int index)
{
    return "";
}

void CXXExecutableModel::getEventIds(std::list<std::string>&) {
}

void CXXExecutableModel::setEventListener(int index,
        rr::EventListenerPtr eventHandler)
{
}

rr::EventListenerPtr CXXExecutableModel::getEventListener(int index)
{
    return rr::EventListenerPtr();
}

double CXXExecutableModel::getFloatingSpeciesAmountRate(int index,
        const double* reactionRates)
{
    return 0;
}

void CXXExecutableModel::reset(int options)
{
}

void CXXExecutableModel::setRandomSeed(int64_t longLongInt)
{
}

int64_t CXXExecutableModel::getRandomSeed()
{
    return 0;
}

double CXXExecutableModel::getRandom()
{
    return 0;
}

uint32_t CXXExecutableModel::getFlags() const
{
    return 0;
}

void CXXExecutableModel::setFlags(uint32_t unsignedInt)
{
}

} /* namespace rrtesting */
