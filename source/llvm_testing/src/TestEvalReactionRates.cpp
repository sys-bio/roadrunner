/*
 * TestEvalReactionRates.cpp
 *
 *  Created on: Jul 20, 2013
 *      Author: andy
 */
#pragma hdrstop
#include "TestEvalReactionRates.h"
#include "rrLogger.h"
#include "rrRoadRunner.h"

namespace rr
{

using namespace std;

TestEvalReactionRates::TestEvalReactionRates(const std::string& compiler,
        const std::string& version, int caseNumber)
: TestBase(compiler, version, caseNumber)
{
}

TestEvalReactionRates::~TestEvalReactionRates()
{
}

bool TestEvalReactionRates::test()
{
    Log(Logger::LOG_INFORMATION) << "Evaluating Initial Conditions for " << fileName << endl;



    Log(Logger::LOG_INFORMATION) << model << endl;

    Log(Logger::LOG_INFORMATION) << "Evaluating Reaction Rates for " << fileName << endl;



    Log(Logger::LOG_INFORMATION) << model << endl;

    return true;
}


void testAmountRates(const char* fname)
{
    RoadRunner r(fname);

    ExecutableModel *model = r.getModel();

    r.simulate();

    int reactions = model->getNumReactions();
    int species = model->getNumFloatingSpecies();

    vector<double> reactionRates(reactions);

    vector<double> amountRates(species);

    model->getReactionRates(reactions, NULL, &reactionRates[0]);

    for (int i = 0; i < reactionRates.size(); ++i)
    {
        cout << "reaction rate " << i << ": " << reactionRates[i] << std::endl;
    }

    for (int i = 0; i < species; ++i)
    {
        double amtRate1;
        model->getFloatingSpeciesAmountRates(1, &i, &amtRate1);
        double amtRate2 = model->getFloatingSpeciesAmountRate(i, &reactionRates[0]);

        cout << "amount rate " << i << ": " << amtRate1 << ", " << amtRate2 << std::endl;
    }
}

void testStoch(const char* fname)
{
    RoadRunner r(fname);

    SimulateOptions o = SimulateOptions();

	r.setIntegrator("gillespie");

    r.simulate(&o);


}



} /* namespace rr */
