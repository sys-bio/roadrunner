/*
 * CXXEnzymeExecutableModel.cpp
 *
 *  Created on: Dec 11, 2014
 *      Author: andy
 */

#include "CXXEnzymeExecutableModel.h"
#include "rrLogger.h"
#include "rrOSSpecifics.h"

#include <stdexcept>
#include <cmath>
#include <cstring>

using rr::Logger;

namespace rrtesting
{


CXXEnzymeExecutableModel::CXXEnzymeExecutableModel(const rr::Dictionary* dict)
{
    Log(Logger::LOG_NOTICE) << __FUNC__;

    volumes = new double[1];

    if (dict->hasKey("numSource")) {
        numSource = dict->getItem("numSource");
    } else {
        numSource = 10;
    }


    numReactions = numSource;
    numSpecies = 2*numSource;

    Log(Logger::LOG_NOTICE) << "creating " << numSpecies << " number of species test";


    floatingSpeciesAmounts = new double[numSpecies];

    reactionRates = new double[numReactions];

    std::vector<unsigned> rowidx;
    std::vector<unsigned> colidx;
    std::vector<double> values;

    for(int r = 0; r < numReactions; ++r) {
        // consume an s
        colidx.push_back(r);
        rowidx.push_back(r);
        values.push_back(-1);

        // produce a p
        colidx.push_back(r);
        rowidx.push_back(r+numSource);
        values.push_back(1);
    }

    stoichiometry = rr::csr_matrix_new(numSpecies, numReactions, rowidx, colidx, values);

    reset(0);
}

CXXEnzymeExecutableModel::~CXXEnzymeExecutableModel()
{
    Log(Logger::LOG_NOTICE) << __FUNC__;

    delete[] volumes;
    delete[] floatingSpeciesAmounts;
    delete[] reactionRates ;
    rr::csr_matrix_delete(stoichiometry);
}

std::string CXXEnzymeExecutableModel::getModelName()
{
    return "C++ Enzyme Test Model";
}

void CXXEnzymeExecutableModel::setTime(double _time)
{
}

double CXXEnzymeExecutableModel::getTime()
{
    return 0;
}

void CXXEnzymeExecutableModel::reset()
{
    reset(0);
}

int CXXEnzymeExecutableModel::getNumDepFloatingSpecies()
{
    return 0;
}

int CXXEnzymeExecutableModel::getNumFloatingSpecies()
{
    return 2 * numSource;
}

int CXXEnzymeExecutableModel::getFloatingSpeciesIndex(const std::string& eid)
{
    return 0;
}

std::string CXXEnzymeExecutableModel::getFloatingSpeciesId(int index)
{
    return "";
}

int CXXEnzymeExecutableModel::getNumIndFloatingSpecies()
{
    return 0;
}

int CXXEnzymeExecutableModel::getFloatingSpeciesAmounts(int len, const int* indx,
        double* values)
{
    double value;
    for (int i = 0; i < len; ++i)
    {
        int j = indx ? indx[i] : i;
        value = floatingSpeciesAmounts[j];


        values[i] = value;
    }
    return len;
}

int CXXEnzymeExecutableModel::setFloatingSpeciesAmounts(int len, const int* indx,
        const double* values)
{
    return 0;
}

int CXXEnzymeExecutableModel::getFloatingSpeciesAmountRates(int len, const int* indx,
        double* values)
{
    return 0;
}

int CXXEnzymeExecutableModel::getFloatingSpeciesConcentrationRates(int len,
        const int* indx, double* values)
{
    return 0;
}

int CXXEnzymeExecutableModel::getFloatingSpeciesConcentrations(int len,
        const int* indx, double* values)
{
    return 0;
}

int CXXEnzymeExecutableModel::setFloatingSpeciesConcentrations(int len,
        const int* indx, const double* values)
{
    return 0;
}

int CXXEnzymeExecutableModel::setFloatingSpeciesInitConcentrations(int len,
        const int* indx, const double* values)
{
    return 0;
}

int CXXEnzymeExecutableModel::getFloatingSpeciesInitConcentrations(int len,
        const int* indx, double* values)
{
    return 0;
}

int CXXEnzymeExecutableModel::setFloatingSpeciesInitAmounts(int len, const int* indx,
        const double* values)
{
    return 0;
}

int CXXEnzymeExecutableModel::getFloatingSpeciesInitAmounts(int len, const int* indx,
        double* values)
{
    return 0;
}

int CXXEnzymeExecutableModel::getNumBoundarySpecies()
{
    return 0;
}

int CXXEnzymeExecutableModel::getBoundarySpeciesIndex(const std::string& eid)
{
    return 0;
}

std::string CXXEnzymeExecutableModel::getBoundarySpeciesId(int index)
{
    return "";
}

int CXXEnzymeExecutableModel::getBoundarySpeciesAmounts(int len, const int* indx,
        double* values)
{
    return 0;
}

int CXXEnzymeExecutableModel::getBoundarySpeciesConcentrations(int len,
        const int* indx, double* values)
{
    return 0;
}

int CXXEnzymeExecutableModel::setBoundarySpeciesConcentrations(int len,
        const int* indx, const double* values)
{
    return 0;
}

int CXXEnzymeExecutableModel::getNumGlobalParameters()
{
    return 0;
}

int CXXEnzymeExecutableModel::getGlobalParameterIndex(const std::string& eid)
{
    return 0;
}

std::string CXXEnzymeExecutableModel::getGlobalParameterId(int index)
{
    return "";
}

int CXXEnzymeExecutableModel::getGlobalParameterValues(int len, const int* indx,
        double* values)
{
    return 0;
}

int CXXEnzymeExecutableModel::setGlobalParameterValues(int len, const int* indx,
        const double* values)
{
    return 0;
}

int CXXEnzymeExecutableModel::getNumCompartments()
{
    return 0;
}

int CXXEnzymeExecutableModel::getCompartmentIndex(const std::string& eid)
{
    return 0;
}

std::string CXXEnzymeExecutableModel::getCompartmentId(int index)
{
    return "";
}

int CXXEnzymeExecutableModel::getCompartmentVolumes(int len, const int* indx,
        double* values)
{
    return 0;
}

int CXXEnzymeExecutableModel::setCompartmentVolumes(int len, const int* indx,
        const double* values)
{
    return 0;
}

int CXXEnzymeExecutableModel::setCompartmentInitVolumes(int len, const int* indx,
        const double* values)
{
    return 0;
}

int CXXEnzymeExecutableModel::getCompartmentInitVolumes(int len, const int* indx,
        double* values)
{
    return 0;
}


/**
 * checks if the bitfield value has all the required flags
 * from type
 */
inline bool checkExact(uint32_t type, uint32_t value) {
    return (value & type) == type;
}


void CXXEnzymeExecutableModel::getIds(int types, std::list<std::string>& ids)
{
    /*
    ids.push_back("Km1");
    ids.push_back("Km2");
    ids.push_back("Km3");
    ids.push_back("sigma");
    ids.push_back("Keq");
    ids.push_back("h");
    ids.push_back("k1");
    ids.push_back("Vm");
    ids.push_back("M");
    ids.push_back("S1");
    ids.push_back("S2");
    ids.push_back("S3");
    ids.push_back("S4");
    ids.push_back("S5");
    ids.push_back("S6");
    ids.push_back("S7");
    ids.push_back("S8");
    ids.push_back("S9");
    ids.push_back("S10");

    ids.push_back("P1");
    ids.push_back("P2");
    ids.push_back("P3");
    ids.push_back("P4");
    ids.push_back("P5");
    ids.push_back("P6");
    ids.push_back("P7");
    ids.push_back("P8");
    ids.push_back("P9");
    ids.push_back("P10");
    */
}

int CXXEnzymeExecutableModel::getSupportedIdTypes()
{
    return 0;
}

double CXXEnzymeExecutableModel::getValue(const std::string& id)
{
    return 0;
}

void CXXEnzymeExecutableModel::setValue(const std::string& id, double value)
{
}

int CXXEnzymeExecutableModel::getStoichiometryMatrix(int* rows, int* cols,
        double** data)
{
    return 0;
}

double CXXEnzymeExecutableModel::getStoichiometry(int speciesIndex, int reactionIndex)
{
    return 0;
}

int CXXEnzymeExecutableModel::getNumConservedMoieties()
{
    return 0;
}

int CXXEnzymeExecutableModel::getConservedMoietyIndex(const std::string& eid)
{
    return 0;
}

std::string CXXEnzymeExecutableModel::getConservedMoietyId(int index)
{
    return "";
}

int CXXEnzymeExecutableModel::getConservedMoietyValues(int len, const int* indx,
        double* values)
{
    return 0;
}

int CXXEnzymeExecutableModel::setConservedMoietyValues(int len, const int* indx,
        const double* values)
{
    return 0;
}

int CXXEnzymeExecutableModel::getNumRateRules()
{
    return 0;
}

int CXXEnzymeExecutableModel::getNumReactions()
{
    return numReactions;
}

int CXXEnzymeExecutableModel::getReactionIndex(const std::string& eid)
{
    return 0;
}

std::string CXXEnzymeExecutableModel::getReactionId(int index)
{
    return "";
}

int CXXEnzymeExecutableModel::getReactionRates(int len, const int* indx,
        double* values)
{

    double conversionFactor = evalReactionRates();


    if (indx)
    {
        for (int i = 0; i < len; ++i)
        {
            int j = indx ? indx[i] : i;
            if (j < numReactions)
            {
                values[i] = reactionRates[j];
            }
            else
            {
                throw std::logic_error("index out of range");
            }
        }
    }
    else
    {
        if (len > numReactions)
        {
            throw std::logic_error("invalid length, length must be <= numReactions");
        }
        std::memcpy(values, reactionRates, len * sizeof(double));
    }
    return len;
}

void CXXEnzymeExecutableModel::getRateRuleValues(double* rateRuleValues)
{
}

std::string CXXEnzymeExecutableModel::getStateVectorId(int index)
{
    return "";
}

int CXXEnzymeExecutableModel::getStateVector(double* stateVector)
{
    if(stateVector != NULL) {
        std::memcpy(stateVector, floatingSpeciesAmounts, numSpecies*sizeof(double));
    }
    return numSpecies;
}

int CXXEnzymeExecutableModel::setStateVector(const double* stateVector)
{
    if(stateVector) {
        std::memcpy(floatingSpeciesAmounts, stateVector, numSpecies*sizeof(double));
    }
    return numSpecies;
}

void CXXEnzymeExecutableModel::getStateVectorRate(double time, const double* y,
        double* dydt)
{
    this->time = time;

    if (y && dydt)
    {
        // save and assign state vector
        double *savedFloatingSpeciesAmounts = floatingSpeciesAmounts;
        floatingSpeciesAmounts = const_cast<double*>(y);
        floatingSpeciesAmountRates = dydt;

        // not setting state vector, react rates get dirty
        double conversionFactor = evalReactionRates();

        csr_matrix_dgemv(conversionFactor, stoichiometry, reactionRates, 0.0, floatingSpeciesAmountRates);

        floatingSpeciesAmounts = savedFloatingSpeciesAmounts;
        floatingSpeciesAmountRates = 0;
    }
    else if (y && !dydt)
    {
        setStateVector(y);
    }
    else if (!y && dydt)
    {
        // evaluate dydt using current state

        double conversionFactor = evalReactionRates();

        // floatingSpeciesAmountRates only valid for the following two
        // functions, this will move to a parameter shortly...

        floatingSpeciesAmountRates = dydt;

        csr_matrix_dgemv(conversionFactor, stoichiometry, reactionRates, 0.0, floatingSpeciesAmountRates);

        floatingSpeciesAmountRates = 0;
    }
}

void CXXEnzymeExecutableModel::testConstraints()
{
}

std::string CXXEnzymeExecutableModel::getInfo()
{
    return "C++ Enzyme Test Model";
}

void CXXEnzymeExecutableModel::print(std::ostream& stream)
{
}

int CXXEnzymeExecutableModel::getNumEvents()
{
    return 0;
}

int CXXEnzymeExecutableModel::getEventTriggers(int len, const int* indx,
        unsigned char* values)
{
    return 0;
}

int CXXEnzymeExecutableModel::applyEvents(double timeEnd,
        const unsigned char* previousEventStatus, const double* initialState,
        double* finalState)
{
    return 0;
}

int CXXEnzymeExecutableModel::applyPendingEvents(double timeEnd)
{
    return 0;
}

void CXXEnzymeExecutableModel::getEventRoots(double time, const double* y,
        double* gdot)
{
}

double CXXEnzymeExecutableModel::getNextPendingEventTime(bool pop)
{
    return 0;
}

int CXXEnzymeExecutableModel::getPendingEventSize()
{
    return 0;
}

void CXXEnzymeExecutableModel::resetEvents()
{
}

int CXXEnzymeExecutableModel::getEventIndex(const std::string& eid)
{
    return 0;
}

std::string CXXEnzymeExecutableModel::getEventId(int index)
{
    return "";
}

void CXXEnzymeExecutableModel::setEventListener(int index,
        rr::EventListenerPtr eventHandler)
{
}

rr::EventListenerPtr CXXEnzymeExecutableModel::getEventListener(int index)
{
    return rr::EventListenerPtr();
}

double CXXEnzymeExecutableModel::getFloatingSpeciesAmountRate(int index,
        const double* reactionRates)
{
    return 0;
}

void CXXEnzymeExecutableModel::reset(int options)
{
    Km1 = 0.45; Km2 = 0.78; Km3 = 5.6;
    sigma = 0.34; Keq = 2.3;
    h = 3; k1 = 0.6;
    Vm = 11.3;
    M = 0.45;

    volumes[0] = 1;

    int i = 0;

    // set the sources to one incerease each time, set products all to zero
    for(; i < numSource; ++i) {
        floatingSpeciesAmounts[i] = i+1;
    }

    for(; i < numSpecies; ++i) {
        floatingSpeciesAmounts[i] = 0;
    }
}

void CXXEnzymeExecutableModel::setRandomSeed(int64_t longLongInt)
{
}

int64_t CXXEnzymeExecutableModel::getRandomSeed()
{
    return 0;
}

double CXXEnzymeExecutableModel::getRandom()
{
    return 0;
}

uint32_t CXXEnzymeExecutableModel::getFlags() const
{
    return 0;
}

void CXXEnzymeExecutableModel::setFlags(uint32_t unsignedInt)
{
}

//function rate(S, P, Km1, Km2, Km3, sigma, Keq, h, k1, Vm, M)
//   (Vm*(S/Km1)*(1 - (P/S)/Keq)*pow((S/Km1 + P/Km2), (h-1))) / \
//   ((1 + pow(M/Km3, h))/(1 + sigma*pow(M/Km3, h)) + pow(S/Km1 + P/Km2, h));
//end

double rate(double S, double P, double Km1, double Km2, double Km3, double sigma, double Keq, double h, double k1, double Vm, double M)    {

    double num = (Vm*(S/Km1)*(1 - (P/S)/Keq)*pow((S/Km1 + P/Km2), (h-1))) ;
    double denom = ((1 + pow(M/Km3, h))/(1 + sigma*pow(M/Km3, h)) + pow(S/Km1 + P/Km2, h));
    double result = num / denom;
    return result;
}

#define REACTION_RATE(S, P) rate((S), (P), Km1, Km2, Km3, sigma, Keq, h, k1, Vm, M)

double CXXEnzymeExecutableModel::evalReactionRates()
{
    for(int i = 0; i < numReactions; ++i) {
        reactionRates[i] = REACTION_RATE(floatingSpeciesAmounts[i]/volumes[0], floatingSpeciesAmounts[numSource + i]/volumes[0]);
    }
    return 1.0;
}

} /* namespace rrtesting */
