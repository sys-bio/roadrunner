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
        rr::Random     mRandom;

    public:
                        Noise(double m = 0, double s = 1);
        double          getNoise(double mean, double sigma);
};

}
#endif
