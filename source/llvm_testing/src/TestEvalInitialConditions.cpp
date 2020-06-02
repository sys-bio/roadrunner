/*
 * TestEvalInitialConditions.cpp
 *
 *  Created on: Jul 20, 2013
 *      Author: andy
 */
#pragma hdrstop
#include "TestEvalInitialConditions.h"
#include "rrLogger.h"



namespace rr
{

using namespace std;

TestEvalInitialConditions::TestEvalInitialConditions(const std::string& compiler,
        const std::string& version, int caseNumber)
    : TestBase(compiler, version, caseNumber)
{
}

TestEvalInitialConditions::~TestEvalInitialConditions()
{
}

bool TestEvalInitialConditions::test()
{
    Log(Logger::LOG_INFORMATION) << "Evaluating Initial Conditions for " << fileName << endl;

    Log(Logger::LOG_INFORMATION) << model << endl;

    for (size_t i = 0; i < model->getNumFloatingSpecies(); i++)
    {
        double amt;
        double conc;
        model->getFloatingSpeciesAmounts(1, &i, &amt);
        model->getFloatingSpeciesConcentrations(1, &i, &conc);

        Log(Logger::LOG_INFORMATION) << "float species: " << i << ", amount: " << amt << ", conc: " << conc << endl;
    }

    for (size_t i = 0; i < model->getNumBoundarySpecies(); i++)
    {
        double amt;
        double conc;
        model->getBoundarySpeciesAmounts(1, &i, &amt);
        model->getBoundarySpeciesConcentrations(1, &i, &conc);

        Log(Logger::LOG_INFORMATION) << "boundary species: " << i << ", amount: " << amt << ", conc: " << conc << endl;
    }


    return true;
}

} /* namespace rr */
