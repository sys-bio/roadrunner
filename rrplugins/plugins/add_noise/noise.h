#ifndef noiseH
#define noiseH
#include "telRandom.h"
//---------------------------------------------------------------------------

namespace addNoise
{

class Noise
{
    private:
        double          mMean;
        double          mSigma;
        tlp::Random*    mRandom;

    public:
                        Noise(double m = 0, double s = 1);
                        ~Noise();
        double          getNoise(double mean, double sigma);
        void            setSeed(unsigned long val);
};

}
#endif
