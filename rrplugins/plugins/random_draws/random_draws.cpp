#pragma hdrstop
#include <time.h>
#include "random_draws.h"

namespace randomDraws
{
    rrc::THostInterface* gHostInterface = NULL;

    RandomDraws::RandomDraws() :
        CPPPlugin("RandomDraws", "Examples"),
        mSeed(0, "Seed", "Seed for each execute (0 to ignore)"),
        mNrOfDraws(10, "NrOfDraws", "Number of draws to make on each execute"),
        mDraws(std::vector<double>(), "Draws", "The draws made by the last execute"),
        mRandom(new tlp::Random((unsigned long)time(NULL)))
    {
        mVersion = "1.0.0";
        mProperties.add(&mSeed);
        mProperties.add(&mNrOfDraws);
        mProperties.add(&mDraws);
        mHint = "Emits draws from the tlp random number generator.";
        mDescription = "Seeds a tlp::Random on each execute and returns draws from it. Used to check that a seed gives reproducible draws from inside a plugin.";
    }

    RandomDraws::~RandomDraws()
    {
        delete mRandom;
    }

    bool RandomDraws::execute(bool inThread)
    {
        unsigned long seed = mSeed.getValue();
        if (seed != 0)
        {
            delete mRandom;
            mRandom = new tlp::Random(seed);
        }

        std::vector<double>& draws = mDraws.getValueReference();
        draws.clear();
        for (int i = 0; i < mNrOfDraws.getValue(); i++)
        {
            draws.push_back(mRandom->next());
        }
        return true;
    }

#ifdef EXPORT_RANDOM_DRAWS
    TLP_DS Plugin* plugins_cc createPlugin()
    {
        return new RandomDraws;
    }

    TLP_DS const char* plugins_cc getImplementationLanguage()
    {
        return "CPP";
    }

    TLP_DS void plugins_cc setHostInterface(rrc::THostInterface* _hostInterface)
    {
        gHostInterface = _hostInterface;
    }
#endif
}
