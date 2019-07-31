#pragma hdrstop
#include <time.h>
#include <math.h>
#include "noise.h"
#include "rrUtils.h"
#include "telRandom.h"
//---------------------------------------------------------------------------
namespace addNoise
{

double gaussNoise(double mean, double sigma, rr::Random& random);

Noise::Noise(double m, double s)
:
mMean(m),
mSigma(s),
mRandom( (unsigned long) time( NULL ) )
{}

double Noise::getNoise(double mean, double sigma)
{
    mMean = mean;
    mSigma = sigma;
    return gaussNoise(mMean, mSigma, mRandom);
}

/* This uses the fact that a Rayleigh-distributed random variable R, with
the probability distribution F(R) = 0 if R < 0 and F(R) =
1 - exp(-R^2/2*sigma^2) if R >= 0, is related to a pair of Gaussian
variables C and D through the transformation C = R * cos(theta) and
D = R * sin(theta), where theta is a uniformly distributed variable
in the interval (0, 2*pi()). From Contemporary Communication Systems
USING MATLAB(R), by John G. Proakis and Masoud Salehi, published by
PWS Publishing Company, 1998, pp 49-50. This is a pretty good book. */
double gaussNoise(double mean, double sigma, rr::Random& random)
{
    double u, r;            /* uniform and Rayleigh random variables */

    /* generate a uniformly distributed random number u between 0 and 1 - 1E-6*/
    u = random.next(); //(double) rand() / RAND_MAX;
    if (u == 1.0)
    {
        u = 0.999999999;
    }

    /* generate a Rayleigh-distributed random number r using u */
    r = sigma * sqrt( 2.0 * log( 1.0 / (1.0 - u) ) );

    /* generate another uniformly-distributed random number u as before*/
    u = random.next(); //(double) rand() / RAND_MAX;
    if (u == 1.0)
    {
        u = 0.999999999;
    }

    /* generate and return a Gaussian-distributed random number using r and u */
    return( (double) (mean + r * cos(2. * 3.14159265358979323846 * u) ) );
}

}
