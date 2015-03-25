/*
 * CXXBrusselatorExecutableModel.cpp
 *
 *  Created on: Dec 11, 2014
 *      Author: andy
 */

#include "CXXBrusselatorExecutableModel.h"
#include "rrLogger.h"
#include "rrOSSpecifics.h"

using rr::Logger;

namespace rrtesting
{

CXXBrusselatorExecutableModel::CXXBrusselatorExecutableModel(const rr::Dictionary* dict)
{
    Log(Logger::LOG_NOTICE) << __FUNC__;
}

CXXBrusselatorExecutableModel::~CXXBrusselatorExecutableModel()
{
    Log(Logger::LOG_NOTICE) << __FUNC__;
}

std::string CXXBrusselatorExecutableModel::getModelName()
{
    return "C++ Test Model";
}

void CXXBrusselatorExecutableModel::setTime(double _time)
{
}

double CXXBrusselatorExecutableModel::getTime()
{
    return 0;
}

void CXXBrusselatorExecutableModel::reset()
{
}

int CXXBrusselatorExecutableModel::getNumDepFloatingSpecies()
{
    return 0;
}

int CXXBrusselatorExecutableModel::getNumFloatingSpecies()
{
    return 0;
}

int CXXBrusselatorExecutableModel::getFloatingSpeciesIndex(const std::string& eid)
{
    return 0;
}

std::string CXXBrusselatorExecutableModel::getFloatingSpeciesId(int index)
{
    return "";
}

int CXXBrusselatorExecutableModel::getNumIndFloatingSpecies()
{
    return 0;
}

int CXXBrusselatorExecutableModel::getFloatingSpeciesAmounts(int len, const int* indx,
        double* values)
{
    return 0;
}

int CXXBrusselatorExecutableModel::setFloatingSpeciesAmounts(int len, const int* indx,
        const double* values)
{
    return 0;
}

int CXXBrusselatorExecutableModel::getFloatingSpeciesAmountRates(int len, const int* indx,
        double* values)
{
    return 0;
}

int CXXBrusselatorExecutableModel::getFloatingSpeciesConcentrationRates(int len,
        const int* indx, double* values)
{
    return 0;
}

int CXXBrusselatorExecutableModel::getFloatingSpeciesConcentrations(int len,
        const int* indx, double* values)
{
    return 0;
}

int CXXBrusselatorExecutableModel::setFloatingSpeciesConcentrations(int len,
        const int* indx, const double* values)
{
    return 0;
}

int CXXBrusselatorExecutableModel::setFloatingSpeciesInitConcentrations(int len,
        const int* indx, const double* values)
{
    return 0;
}

int CXXBrusselatorExecutableModel::getFloatingSpeciesInitConcentrations(int len,
        const int* indx, double* values)
{
    return 0;
}

int CXXBrusselatorExecutableModel::setFloatingSpeciesInitAmounts(int len, const int* indx,
        const double* values)
{
    return 0;
}

int CXXBrusselatorExecutableModel::getFloatingSpeciesInitAmounts(int len, const int* indx,
        double* values)
{
    return 0;
}

int CXXBrusselatorExecutableModel::getNumBoundarySpecies()
{
    return 0;
}

int CXXBrusselatorExecutableModel::getBoundarySpeciesIndex(const std::string& eid)
{
    return 0;
}

std::string CXXBrusselatorExecutableModel::getBoundarySpeciesId(int index)
{
    return "";
}

int CXXBrusselatorExecutableModel::getBoundarySpeciesAmounts(int len, const int* indx,
        double* values)
{
    return 0;
}

int CXXBrusselatorExecutableModel::getBoundarySpeciesConcentrations(int len,
        const int* indx, double* values)
{
    return 0;
}

int CXXBrusselatorExecutableModel::setBoundarySpeciesConcentrations(int len,
        const int* indx, const double* values)
{
    return 0;
}

int CXXBrusselatorExecutableModel::getNumGlobalParameters()
{
    return 0;
}

int CXXBrusselatorExecutableModel::getGlobalParameterIndex(const std::string& eid)
{
    return 0;
}

std::string CXXBrusselatorExecutableModel::getGlobalParameterId(int index)
{
    return "";
}

int CXXBrusselatorExecutableModel::getGlobalParameterValues(int len, const int* indx,
        double* values)
{
    return 0;
}

int CXXBrusselatorExecutableModel::setGlobalParameterValues(int len, const int* indx,
        const double* values)
{
    return 0;
}

int CXXBrusselatorExecutableModel::getNumCompartments()
{
    return 0;
}

int CXXBrusselatorExecutableModel::getCompartmentIndex(const std::string& eid)
{
    return 0;
}

std::string CXXBrusselatorExecutableModel::getCompartmentId(int index)
{
    return "";
}

int CXXBrusselatorExecutableModel::getCompartmentVolumes(int len, const int* indx,
        double* values)
{
    return 0;
}

int CXXBrusselatorExecutableModel::setCompartmentVolumes(int len, const int* indx,
        const double* values)
{
    return 0;
}

int CXXBrusselatorExecutableModel::setCompartmentInitVolumes(int len, const int* indx,
        const double* values)
{
    return 0;
}

int CXXBrusselatorExecutableModel::getCompartmentInitVolumes(int len, const int* indx,
        double* values)
{
    return 0;
}

void CXXBrusselatorExecutableModel::getIds(int types, std::list<std::string>& ids)
{
}

int CXXBrusselatorExecutableModel::getSupportedIdTypes()
{
    return 0;
}

double CXXBrusselatorExecutableModel::getValue(const std::string& id)
{
    return 0;
}

void CXXBrusselatorExecutableModel::setValue(const std::string& id, double value)
{
}

int CXXBrusselatorExecutableModel::getStoichiometryMatrix(int* rows, int* cols,
        double** data)
{
    return 0;
}

double CXXBrusselatorExecutableModel::getStoichiometry(int speciesIndex, int reactionIndex)
{
    return 0;
}

int CXXBrusselatorExecutableModel::getNumConservedMoieties()
{
    return 0;
}

int CXXBrusselatorExecutableModel::getConservedMoietyIndex(const std::string& eid)
{
    return 0;
}

std::string CXXBrusselatorExecutableModel::getConservedMoietyId(int index)
{
    return "";
}

int CXXBrusselatorExecutableModel::getConservedMoietyValues(int len, const int* indx,
        double* values)
{
    return 0;
}

int CXXBrusselatorExecutableModel::setConservedMoietyValues(int len, const int* indx,
        const double* values)
{
    return 0;
}

int CXXBrusselatorExecutableModel::getNumRateRules()
{
    return 0;
}

int CXXBrusselatorExecutableModel::getNumReactions()
{
    return 0;
}

int CXXBrusselatorExecutableModel::getReactionIndex(const std::string& eid)
{
    return 0;
}

std::string CXXBrusselatorExecutableModel::getReactionId(int index)
{
    return "";
}

int CXXBrusselatorExecutableModel::getReactionRates(int len, const int* indx,
        double* values)
{
    return 0;
}

void CXXBrusselatorExecutableModel::getRateRuleValues(double* rateRuleValues)
{
}

std::string CXXBrusselatorExecutableModel::getStateVectorId(int index)
{
    return "";
}

int CXXBrusselatorExecutableModel::getStateVector(double* stateVector)
{
    return 0;
}

int CXXBrusselatorExecutableModel::setStateVector(const double* stateVector)
{
    return 0;
}

void CXXBrusselatorExecutableModel::getStateVectorRate(double time, const double* y,
        double* dydt)
{
}

void CXXBrusselatorExecutableModel::testConstraints()
{
}

std::string CXXBrusselatorExecutableModel::getInfo()
{
    return "C++ Test Model";
}

void CXXBrusselatorExecutableModel::print(std::ostream& stream)
{
}

int CXXBrusselatorExecutableModel::getNumEvents()
{
    return 0;
}

int CXXBrusselatorExecutableModel::getEventTriggers(int len, const int* indx,
        unsigned char* values)
{
    return 0;
}

int CXXBrusselatorExecutableModel::applyEvents(double timeEnd,
        const unsigned char* previousEventStatus, const double* initialState,
        double* finalState)
{
    return 0;
}

int CXXBrusselatorExecutableModel::applyPendingEvents(double timeEnd)
{
    return 0;
}

void CXXBrusselatorExecutableModel::getEventRoots(double time, const double* y,
        double* gdot)
{
}

double CXXBrusselatorExecutableModel::getNextPendingEventTime(bool pop)
{
    return 0;
}

int CXXBrusselatorExecutableModel::getPendingEventSize()
{
    return 0;
}

void CXXBrusselatorExecutableModel::resetEvents()
{
}

int CXXBrusselatorExecutableModel::getEventIndex(const std::string& eid)
{
    return 0;
}

std::string CXXBrusselatorExecutableModel::getEventId(int index)
{
    return "";
}

void CXXBrusselatorExecutableModel::setEventListener(int index,
        rr::EventListenerPtr eventHandler)
{
}

rr::EventListenerPtr CXXBrusselatorExecutableModel::getEventListener(int index)
{
    return rr::EventListenerPtr();
}

double CXXBrusselatorExecutableModel::getFloatingSpeciesAmountRate(int index,
        const double* reactionRates)
{
    return 0;
}

void CXXBrusselatorExecutableModel::reset(int options)
{
}

void CXXBrusselatorExecutableModel::setRandomSeed(int64_t longLongInt)
{
}

int64_t CXXBrusselatorExecutableModel::getRandomSeed()
{
    return 0;
}

double CXXBrusselatorExecutableModel::getRandom()
{
    return 0;
}

uint32_t CXXBrusselatorExecutableModel::getFlags() const
{
    return 0;
}

void CXXBrusselatorExecutableModel::setFlags(uint32_t unsignedInt)
{
}

} /* namespace rrtesting */
