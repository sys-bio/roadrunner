#ifndef randomDrawsH
#define randomDrawsH
#include <vector>
#include "telProperty.h"
#include "telRandom.h"
#include "rrplugins/pluginBaseClass/telCPPPlugin.h"
#include "rrplugins/core/tel_api.h"

namespace randomDraws
{
    using namespace tlp;

    //Seeds a tlp::Random the way the Monte Carlo plugin does and emits draws from it.
    class RandomDraws : public CPPPlugin
    {
    public:
        RandomDraws();
        ~RandomDraws();
        bool                        execute(bool inThread = false);

    private:
        Property<unsigned long>     mSeed;
        Property<int>               mNrOfDraws;
        Property< std::vector<double> > mDraws;
        tlp::Random*                mRandom;
    };

#ifdef EXPORT_RANDOM_DRAWS
    extern "C"
    {
        TLP_DS Plugin*      plugins_cc createPlugin();
        TLP_DS const char*  plugins_cc getImplementationLanguage();
        TLP_DS void         plugins_cc setHostInterface(rrc::THostInterface* _hostInterface);
    }
#endif
}

#endif //randomDrawsH
