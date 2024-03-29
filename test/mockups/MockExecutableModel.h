//
// Created by Ciaran on 19/05/2021.
//

#ifndef ROADRUNNER_MOCKEXECUTABLEMODEL_H
#define ROADRUNNER_MOCKEXECUTABLEMODEL_H

#include "gmock/gmock.h"
#include "rrExecutableModel.h"

using namespace rr;

class MockExecutableModel : public rr::ExecutableModel{
public:
    MOCK_METHOD(std::string, getExecutableModelDesc, (), (const, override));
    MOCK_METHOD(std::string, getModelName, (), (override));
    MOCK_METHOD(void, setTime, (double _time), (override));
    MOCK_METHOD(double, getTime, (), (override));
    MOCK_METHOD(void, reset, (), (override));
    MOCK_METHOD(int, getNumDepFloatingSpecies, (), (override));
    MOCK_METHOD(int, getNumFloatingSpecies, (), (override));
    MOCK_METHOD(int, getFloatingSpeciesIndex, (const std::string &eid), (override));
    MOCK_METHOD(std::string, getFloatingSpeciesId, (size_t index), (override));
    MOCK_METHOD(int, getNumIndFloatingSpecies, (), (override));
    MOCK_METHOD(int, getFloatingSpeciesAmounts, (size_t len, int const *indx, double *values), (override));
    MOCK_METHOD(int, setFloatingSpeciesAmounts, (size_t len, int const *indx, const double *values), (override));
    MOCK_METHOD(int, setFloatingSpeciesAmounts, (size_t len, int const* indx, const double* values, bool strict), (override));
    MOCK_METHOD(int, getFloatingSpeciesAmountRates, (size_t len, int const *indx, double *values), (override));
    MOCK_METHOD(int, getFloatingSpeciesConcentrationRates, (size_t len, int const *indx, double *values), (override));
    MOCK_METHOD(int, getFloatingSpeciesConcentrations, (size_t len, int const *indx, double *values), (override));
    MOCK_METHOD(int, setFloatingSpeciesConcentrations, (size_t len, int const *indx, double const *values), (override));
    MOCK_METHOD(int, setFloatingSpeciesInitConcentrations, (size_t len, int const *indx, double const *values), (override));
    MOCK_METHOD(int, getFloatingSpeciesInitConcentrations, (size_t len, int const *indx, double *values), (override));
    MOCK_METHOD(int, setFloatingSpeciesInitAmounts, (size_t len, int const *indx, double const *values), (override));
    MOCK_METHOD(int, getFloatingSpeciesInitAmounts, (size_t len, int const *indx, double *values), (override));
    MOCK_METHOD(int, getNumBoundarySpecies, (), (override));
    MOCK_METHOD(int, getBoundarySpeciesIndex, (const std::string &eid), (override));
    MOCK_METHOD(std::string, getBoundarySpeciesId, (size_t index), (override));
    MOCK_METHOD(int, getBoundarySpeciesAmounts,            (size_t len, int const *indx, double *values),        (override));
    MOCK_METHOD(int, getBoundarySpeciesConcentrations,     (size_t len, int const *indx, double *values),        (override));
    MOCK_METHOD(int, setBoundarySpeciesConcentrations,     (size_t len, int const *indx, double const *values), (override));
    MOCK_METHOD(int, setBoundarySpeciesAmounts, (size_t len, int const* indx, double const* values), (override));
    MOCK_METHOD(int, setBoundarySpeciesAmounts, (size_t len, int const* indx, double const* values, bool strict), (override));
    MOCK_METHOD(int, setBoundarySpeciesInitConcentrations, (size_t len, int const* indx, double const* values), (override));
    MOCK_METHOD(int, getBoundarySpeciesInitConcentrations, (size_t len, int const* indx, double* values), (override));
    MOCK_METHOD(int, setBoundarySpeciesInitAmounts, (size_t len, int const* indx, double const* values), (override));
    MOCK_METHOD(int, getBoundarySpeciesInitAmounts, (size_t len, int const* indx, double* values), (override));
    MOCK_METHOD(int, getNumGlobalParameters,               (), (override));
    MOCK_METHOD(int, getGlobalParameterIndex,              (const std::string &eid), (override));
    MOCK_METHOD(std::string, getGlobalParameterId, (size_t index), (override));
    MOCK_METHOD(int, getGlobalParameterValues, (size_t len, int const *indx, double *values), (override));
    MOCK_METHOD(int, setGlobalParameterValues, (size_t len, int const *indx, const double *values), (override));
    MOCK_METHOD(int, setGlobalParameterValues, (size_t len, int const* indx, const double* values, bool strict), (override));
    MOCK_METHOD(int, setGlobalParameterInitValues, (size_t len, int const *indx, double const *values), (override));
    MOCK_METHOD(int, getGlobalParameterInitValues, (size_t len, int const *indx, double *values), (override));
    MOCK_METHOD(int, getNumCompartments, (), (override));
    MOCK_METHOD(int, getCompartmentIndexForFloatingSpecies, (size_t index), (override));
    MOCK_METHOD(int, getCompartmentIndexForBoundarySpecies, (size_t index), (override));
    MOCK_METHOD(int, getCompartmentIndex, (const std::string &eid), (override));
    MOCK_METHOD(std::string, getCompartmentId, (size_t index), (override));
    MOCK_METHOD(int, getCompartmentVolumes, (size_t len, int const *indx, double *values), (override));
    MOCK_METHOD(int, setCompartmentVolumes, (size_t len, int const *indx, const double *values), (override));
    MOCK_METHOD(int, setCompartmentVolumes, (size_t len, int const* indx, const double* values, bool strict), (override));
    MOCK_METHOD(int, setCompartmentInitVolumes, (size_t len, int const *indx, double const *values), (override));
    MOCK_METHOD(int, getCompartmentInitVolumes, (size_t len, int const *indx, double *values), (override));
    MOCK_METHOD(void, getIds, (int types, std::list<std::string> & ids), (override));
    MOCK_METHOD(int, getSupportedIdTypes, (), (override));
    MOCK_METHOD(double, getValue, (const std::string &id), (override));
    MOCK_METHOD(void, setValue, (const std::string &id, double value), (override));
    MOCK_METHOD(int, setStoichiometries, (size_t len, int const *indx, const double *values), (override));
    MOCK_METHOD(int, setStoichiometries, (size_t len, int const* indx, const double* values, bool strict), (override));
    MOCK_METHOD(int, setStoichiometry, (int index, double value), (override));
    MOCK_METHOD(int, setStoichiometry, (int speciesIndex, int reactionIndex, double value), (override));
    MOCK_METHOD(double, getStoichiometry, (int index), (override));
    MOCK_METHOD(double, getStoichiometry, (int speciesIndex, int reactionIndex), (override));
    MOCK_METHOD(int, getNumConservedMoieties, (), (override));
    MOCK_METHOD(int, getConservedMoietyIndex, (const std::string &eid), (override));
    MOCK_METHOD(std::string, getConservedMoietyId, (size_t index), (override));
    MOCK_METHOD(int, getConservedMoietyValues, (size_t len, int const *indx, double *values), (override));
    MOCK_METHOD(int, setConservedMoietyValues, (size_t len, int const *indx, const double *values), (override));
    MOCK_METHOD(int, getNumRateRules, (), (override));
    MOCK_METHOD(std::vector<std::string>, getRateRuleSymbols, (), (const, override));
    MOCK_METHOD(int, getNumReactions, (), (override));
    MOCK_METHOD(int, getReactionIndex, (const std::string &eid), (override));
    MOCK_METHOD(std::string, getReactionId, (size_t index), (override));
    MOCK_METHOD(int, getStoichiometryIndex, (const std::string &eid), (override));
    MOCK_METHOD(int, getStoichiometryIndex, (const std::string &sid, const std::string &rid), (override));
    MOCK_METHOD(std::string, getStoichiometryId, (size_t index), (override));
    MOCK_METHOD(int, getReactionRates, (size_t len, int const *indx, double *values), (override));
    MOCK_METHOD(void, getRateRuleValues, (double * rateRuleValues), (override));
    MOCK_METHOD(std::string, getStateVectorId, (size_t index), (override));
    MOCK_METHOD(int, getStateVector, (double * stateVector), (override));
    MOCK_METHOD(int, setStateVector, (const double *stateVector), (override));
    MOCK_METHOD(void, getStateVectorRate, (double time, const double *y, double *dydt), (override));
    MOCK_METHOD(void, testConstraints, (), (override));
    MOCK_METHOD(std::string, getInfo, (), (override));
    MOCK_METHOD(void, print, (std::ostream & stream), (override));
    MOCK_METHOD(int, getNumEvents, (), (override));
    MOCK_METHOD(int, getNumPiecewiseTriggers, (), (override));
    MOCK_METHOD(int, getEventTriggers, (size_t len, const int *indx, unsigned char *values), (override));
    MOCK_METHOD(int, applyEvents,
                (double timeEnd, const unsigned char *previousEventStatus,const double *initialState, double *finalState),
                (override));
    MOCK_METHOD(void, getEventRoots, (double time, const double *y, double *gdot), (override));
    MOCK_METHOD(void, getPiecewiseTriggerRoots, (double time, const double* y, double* gdot), (override));
    MOCK_METHOD(double, getNextPendingEventTime, (bool pop), (override));
    MOCK_METHOD(int, getPendingEventSize, (), (override));
    MOCK_METHOD(void, resetEvents, (), (override));
    MOCK_METHOD(int, getEventIndex, (const std::string &eid), (override));
    MOCK_METHOD(std::string, getEventId, (size_t index), (override));
    MOCK_METHOD(void, getEventIds, (std::list<std::string> & ), (override));
    MOCK_METHOD(void, getAssignmentRuleIds, (std::list<std::string>&), (override));
    MOCK_METHOD(void, getRateRuleIds, (std::list<std::string>&), (override));
    MOCK_METHOD(void, getInitialAssignmentIds, (std::list<std::string>&), (override));
    MOCK_METHOD(void, setEventListener, (size_t index, EventListenerPtr eventHandler), (override));
    MOCK_METHOD(EventListenerPtr, getEventListener, (size_t index), (override));
    MOCK_METHOD(double, getFloatingSpeciesAmountRate, (size_t index,const double *reactionRates), (override));
    MOCK_METHOD(void, reset, (int options), (override));
    MOCK_METHOD(void, setRandomSeed, (int64_t), (override));
    MOCK_METHOD(int64_t, getRandomSeed, (), (override));
    MOCK_METHOD(double, getRandom, (), (override));
    MOCK_METHOD(uint32_t, getFlags, (), (const, override));
    MOCK_METHOD(void, setFlags, (uint32_t), (override));
    MOCK_METHOD(void, computeAllRatesOfChange, (), ());
    MOCK_METHOD(void, saveState, (std::ostream & out), (override));

};


#endif //ROADRUNNER_MOCKEXECUTABLEMODEL_H
