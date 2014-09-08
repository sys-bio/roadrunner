// == PREAMBLE ================================================

// * Licensed under the Apache License, Version 2.0; see README

/*
 * GPUSimBiochemModel.cpp
 *
 *  Created on: Aug 13, 2014
 *      Author: JKM
 */

// == INCLUDES ================================================

#include "GPUSimBiochemModel.h"
#include "GPUSimModelGenerator.h"
#include "rrConfig.h"
// #include "nullptr.h"

// == CODE ====================================================

namespace rr
{

namespace rrgpu
{

// --* ModelElement *--

ModelElement::~ModelElement() {}

// --* GPUSimModel *--

FloatingSpecies* GPUSimModel::getFloatingSpeciesById(const std::string& id) {
    for(FloatingSpecies* s : getFloatingSpecies())
        if(s->matchId(id))
            return s;
    throw_gpusim_exception("No such floating species for id \"" + id + "\"");
}

const FloatingSpecies* GPUSimModel::getFloatingSpeciesById(const std::string& id) const {
    for(const FloatingSpecies* s : getFloatingSpecies())
        if(s->matchId(id))
            return s;
    throw_gpusim_exception("No such floating species for id \"" + id + "\"");
}

Species* GPUSimModel::getSpeciesById(const std::string& id) {
    for(FloatingSpecies* s : getFloatingSpecies())
        if(s->matchId(id))
            return s;
    for(BoundarySpecies* s : getBoundarySpecies())
        if(s->matchId(id))
            return s;
    throw_gpusim_exception("No such species for id \"" + id + "\"");
}

const Species* GPUSimModel::getSpeciesById(const std::string& id) const {
    for(const FloatingSpecies* s : getFloatingSpecies())
        if(s->matchId(id))
            return s;
    for(BoundarySpecies* s : getBoundarySpecies())
        if(s->matchId(id))
            return s;
    throw_gpusim_exception("No such species for id \"" + id + "\"");
}

bool GPUSimModel::hasInitialAssignmentRule(const FloatingSpecies* s) {
    for(const ModelRule* r : getRules())
        if (r->isInitialAssignmentRule() && r->contains(s))
            return true;
    return false;
}

bool GPUSimModel::hasAssignmentRule(const FloatingSpecies* s) const {
    for(const ModelRule* r : getRules())
        if (r->isAssignmentRule() && r->contains(s))
            return true;
    return false;
}

GPUSimModel::size_type GPUSimModel::getStateVecSize() const {
    return getNumIndepFloatingSpecies();
}

int GPUSimModel::getStateVecComponent(const FloatingSpecies* q) const {
    int n=0;
    for(const FloatingSpecies* s : getFloatingSpecies()) {
        if (s == q) {
            if (!s->getIsIndependent())
                throw_gpusim_exception("Sanity check failed: Floating species \"" + s->getId() + "\" is not independent (hence not part of the state vector)");
            return n;
        }
        if (s->getIsIndependent())
            ++n;
    }
    throw_gpusim_exception("No such floating species \"" + q->getId() + "\"");
}

const FloatingSpecies* GPUSimModel::getFloatingSpeciesFromSVComponent(int i) const {
    int n=0;
    for(const FloatingSpecies* s : getFloatingSpecies()) {
        if (n == i) {
            if (!s->getIsIndependent())
                throw_gpusim_exception("Sanity check failed: Floating species \"" + s->getId() + "\" is not independent (hence not part of the state vector)");
            return s;
        }
        if (s->getIsIndependent())
            ++n;
    }
    throw_gpusim_exception("No such floating species for state vec component " + std::to_string(i) + "");
}

void GPUSimModel::dumpStateVecAssignments() const {
    Log(Logger::LOG_DEBUG) << "State vector assignments";
    int n=0;
    for(const FloatingSpecies* s : getFloatingSpecies()) {
        if (s->getIsIndependent()) {
            Log(Logger::LOG_DEBUG) << n++ << ": " << s->getId();
        }
    }
}

GPUSimModel::ReactionSide GPUSimModel::getReactionSide(const Reaction* r, const FloatingSpecies* s) const {
    return r->getSide(s);
}

void GPUSimModel::getIds(int types, std::list<std::string> &ids) {
//     Log(Logger::LOG_DEBUG) << "GPUSimModel::getIds\n";
    for(const ModelElement* e : getElements()) {
        if(e->matchesType(types)) {
          Log(Logger::LOG_DEBUG) << e->getId() << "\n";
            ids.push_back(e->getId());
        }
    }
}

} // namespace rrgpu

} // namespace rr
