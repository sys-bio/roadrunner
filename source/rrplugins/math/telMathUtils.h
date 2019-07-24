#ifndef telMathUtilsH
#define telMathUtilsH
#include <vector>
#include "telMathExporter.h"
#include "../common/telTelluriumData.h"
#include "telRandom.h"
//---------------------------------------------------------------------------

namespace rr
{

using std::vector;

MATH_DECLSPEC double                getEuclideanNorm(const vector<double>& data);
MATH_DECLSPEC TelluriumData         getNormalProbabilityPlot(const TelluriumData& stdPops);

MATH_DECLSPEC TelluriumData         getStandardizedPopulations(const TelluriumData& population);

MATH_DECLSPEC TelluriumData         getDataSet(int expNr, const TelluriumData& data);

MATH_DECLSPEC vector<double>        getStandardDeviations(const TelluriumData& population);
MATH_DECLSPEC vector<double>        getMeans(const TelluriumData& residualsData);
MATH_DECLSPEC vector<double>        getValuesInColumn(int col, const TelluriumData& data);

MATH_DECLSPEC vector<double>        getWeightValuesInColumn(int col, const TelluriumData& data);
MATH_DECLSPEC double                normsdist(const double x);
MATH_DECLSPEC double                invnormsdist(const double y); // 0 < y < 1;

//Vector double arguments
MATH_DECLSPEC double                getChiSquare(const vector<double>& O, const vector<double>& E, const vector<double>& variances);
MATH_DECLSPEC double                getStandardDeviation(const vector<double>& population, double* theMean = NULL);
MATH_DECLSPEC double                getMean(const vector<double>& population);
MATH_DECLSPEC double                erf(double x);
MATH_DECLSPEC double                cdf(double n);

//Random stuff
MATH_DECLSPEC double                getRandomElement(const vector<double>& vec, Random& random);
}
#endif
