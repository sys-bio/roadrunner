#pragma hdrstop
#include <math.h>
#include <algorithm>
#include "telMathUtils.h"
#include "rrLogger.h"
#include "rrUtils.h"
#include "rrException.h"

#define LM_MACHEP     DBL_EPSILON   /* resolution of arithmetic */
#define LM_DWARF      DBL_MIN       /* smallest nonzero number */
#define LM_SQRT_DWARF sqrt(DBL_MIN) /* square should not underflow */
#define LM_SQRT_GIANT sqrt(DBL_MAX) /* square should not overflow */
#define LM_USERTOL    30*LM_MACHEP  /* users are recommended to require this */
#define SQR(x)   (x)*(x)

namespace rr
{
using namespace std;

TelluriumData  getDataSet(int expNr, const TelluriumData& data)
{
    ArrayedParameter para = data.getArrayedParameter();

    if(para.getNumberOfIncrements() < 1)
    {
        return data;
    }

    int nrOfSets = para.getNumberOfIncrements() + 1;
    bool isFirstColTime = data.isFirstColumnTime();
    int nrCols = data.cSize();

    int expSize = (nrCols - (isFirstColTime ? 1 : 0) )/ nrOfSets;
    int startCol = expNr * expSize - (isFirstColTime ? 1 : 0);

    //If a data set contain several experiments, extract a single one
    TelluriumData dataSet(data.rSize(), expSize + (isFirstColTime ? 1 : 0) );
    StringList colHeaders;

    if(isFirstColTime)
    {
        colHeaders.add("Time");
    }

    for(int col = (isFirstColTime ? 1 : 0); col < expSize + 1; col++)
    {
        for(int row = 0; row < data.rSize(); row++)
        {
            dataSet(row, col) = data(row, startCol);
        }
        colHeaders.add(data.getColumnName(startCol));
        startCol++;
    }

    //And time, if any
    if(isFirstColTime)
    {
        for(int row = 0; row < data.rSize(); row++)
        {
            dataSet(row, 0) = data(row, 0);
        }
    }

    //Column header
    dataSet.setColumnNames(colHeaders);
    return dataSet;
}

double getEuclideanNorm(const vector<double>& x)
{
/*     Given an n-vector x, this function calculates the
 *     euclidean norm of x.
 *
 *     The euclidean norm is computed by accumulating the sum of
 *     squares in three different sums. The sums of squares for the
 *     small and large components are scaled so that no overflows
 *     occur. Non-destructive underflows are permitted. Underflows
 *     and overflows do not occur in the computation of the unscaled
 *     sum of squares for the intermediate components.
 *     The definitions of small, intermediate and large components
 *     depend on two constants, LM_SQRT_DWARF and LM_SQRT_GIANT. The main
 *     restrictions on these constants are that LM_SQRT_DWARF**2 not
 *     underflow and LM_SQRT_GIANT**2 not overflow.
 *
 *     Parameters
 *
 *      n is a positive integer input variable.
 *
 *      x is an input array of length n.
 */
    int i;
    double agiant, s1, s2, s3, xabs, x1max, x3max, temp;

    s1 = 0;
    s2 = 0;
    s3 = 0;
    x1max = 0;
    x3max = 0;
    unsigned int n = x.size();
    agiant = LM_SQRT_GIANT / n;

    /** sum squares. **/

    for (i = 0; i < n; i++) {
        xabs = fabs(x[i]);
        if (xabs > LM_SQRT_DWARF) {
            if ( xabs < agiant ) {
                s2 += xabs * xabs;
            } else if ( xabs > x1max ) {
                temp = x1max / xabs;
                s1 = 1 + s1 * SQR(temp);
                x1max = xabs;
            } else {
                temp = xabs / x1max;
                s1 += SQR(temp);
            }
        } else if ( xabs > x3max ) {
            temp = x3max / xabs;
            s3 = 1 + s3 * SQR(temp);
            x3max = xabs;
        } else if (xabs != 0.) {
            temp = xabs / x3max;
            s3 += SQR(temp);
        }
    }

    /** calculation of norm. **/

    if (s1 != 0)
        return x1max * sqrt(s1 + (s2 / x1max) / x1max);
    else if (s2 != 0)
        if (s2 >= x3max)
            return sqrt(s2 * (1 + (x3max / s2) * (x3max * s3)));
        else
            return sqrt(x3max * ((s2 / x3max) + (x3max * s3)));
    else
        return x3max * sqrt(s3);

}

double getChiSquare(const vector<double>& O, const vector<double>& E, const vector<double>& variances)
{
    stringstream msg;
    if (O.size() != E.size() || O.size() != variances.size())
    {
        msg<< "Non equally sized data passed to getChiSquare\n";
        msg<<"Experimental data size is: "<<O.size()<<endl;
        msg<<"Model data size is: "<<E.size()<<endl;
        msg<<"Variances data size is: "<<variances.size()<<endl;
        throw(Exception(msg.str()));
    }

    double chiSquare = 0;

    for(int i = 0; i < O.size(); i++)
    {
        if(variances[i] != 0)
        {
            chiSquare += (1./variances[i]) * pow(O[i] - E[i], 2);
        }
        else
        {
            Log(lError)<<"Tried to divide by zero in gtChiSquare()";
        }
    }
    return chiSquare;
}

TelluriumData getNormalProbabilityPlot(const TelluriumData& stdPops)
{
    bool timeIsFirstCol = stdPops.isFirstColumnTime();
    //Resulting data contain the probPlot for specie1 in col 1 and 2, specie2 in col 3,4 etc..
    TelluriumData probPlots(stdPops.rSize(), 2*(stdPops.cSize() - ((timeIsFirstCol == true) ? 1 : 0)) );

    for(int col = ((timeIsFirstCol == true) ? 1 : 0); col < stdPops.cSize(); col++)
    {
        vector<double> temp = getValuesInColumn(col, stdPops);
        //Sort the values
        sort(temp.begin(), temp.end());

        //Get percentiles
        vector<double> p(temp.size());
        for(int i = 0; i < temp.size(); i++)
        {
             p[i] = ((i+1) - 0.5) / temp.size();
        }

        //Get theoretical z scores from percentiles ?
        vector<double> z(temp.size());
        for(int i = 0; i < temp.size(); i++)
        {            
            z[i] = invnormsdist(p[i]);
        }

        int correction = (timeIsFirstCol == true) ? 1 : 0;
        int popStart = 2*(col - correction);
        //Populate propbPlots with the sorted values and z's
        for(int row = 0; row < temp.size(); row++)
        {
            //X and Y
            probPlots(row, popStart)        = z[row];
            probPlots(row, popStart + 1)    = temp[row];
        }
     }
    return probPlots;
}

//The normsdist and innormsdist is from
//http://mathforum.org/kb/message.jspa?messageID=1525629
//---------------------------------------------------------------------------
// NORMSDIST
// Approximation of normal density function, solves for area under the
// standard normal curve from -infinity to x.
// Duplicates MS Excel NORMSDIST() function, and agrees to within 1e-10
//---------------------------------------------------------------------------
double normsdist(const double x)
{
    static const double invsqrt2pi = 0.3989422804014327, // 1/sqrt(2*pi)
    b1 = 0.31938153,
    b2 = -0.356563782,
    b3 = 1.781477937,
    b4 = -1.821255978,
    b5 = 1.330274429,
    p = 0.2316419;
    double t1, t2, t3, t4, t5, area, xx = fabs(x);
    t1 = 1.0 / (1.0 + p * xx);
    t2 = t1 * t1;
    t3 = t2 * t1;
    t4 = t3 * t1;
    t5 = t4 * t1;
    area = invsqrt2pi * exp(-0.5 * xx*xx)
    * (b1*t1 + b2*t2 + b3*t3 + b4*t4 +b5*t5);
    if (x > 0.0)
        area = 1.0 - area;
    return area;
}

//---------------------------------------------------------------------------
// INVNORMSDIST
// Inverse normal density function, solves for number of standard deviations
// x corresponding to area (probability) y. Duplicates MS Excel NORMSINV().
//---------------------------------------------------------------------------
double invnormsdist(const double y) // 0 < y < 1;
{
    double x, tst, incr;
    if (y < 1.0e-20)
        return -5.0;
    if (y >= 1.0)
        return 5.0;
    x = 0.0;
    incr = y - 0.5;
    while (fabs(incr) > 0.0000001)
    {
        if (fabs(incr) < 0.0001 && (x <= -5.0 || x >= 5.0))
            break;
        x += incr;
        tst = normsdist(x);

    if ((tst > y && incr > 0.) || (tst < y && incr < 0.))
        incr *= -0.5;
    }
    return x;
}

vector<double> getValuesInColumn(int col, const TelluriumData& data)
{
    vector<double> vals;
    if(col < data.cSize())
    {
        vals.resize(data.rSize());
        for(int row = 0; row < data.rSize(); row++)
        {
            vals[row] = data(row, col);
        }
    }
    return vals;
}

double getRandomElement(const vector<double>& vec, Random& random)
{
    int maxIndex = vec.size() -1;
    double rnd = random.next() * maxIndex;
    rnd  = rnd  + 0.5;
    return vec[(int) rnd ];
}

vector<double> getWeightValuesInColumn(int col, const TelluriumData& data)
{
    vector<double> vals;
    if(!data.hasWeights()) //Return weights of '1'
    {
        Log(lDebug3) <<"Trying to read non-existent weight values from data";
    }

    if(col < data.cSize())
    {
        vals.resize(data.rSize());
        for(int row = 0; row < data.rSize(); row++)
        {
            if(data.hasWeights())
            {
                vals[row] = data.getWeight(row, col);
            }
            else
            {
                vals[row] = 1;
            }
        }
    }
    return vals;
}

TelluriumData getStandardizedPopulations(const TelluriumData& population)
{
    TelluriumData stdPop(population.rSize(), population.cSize());
    stdPop.setColumnNames(population.getColumnNames());
    vector<double> stdDeviations = getStandardDeviations(population);

    bool timeIsFirstCol = population.isFirstColumnTime();

    for(int col =0; col <population.cSize(); col++)
    {
        int nonTimeDataIndex = (col - (timeIsFirstCol == true)) ? 1 : 0;

        for(int dataPtn = 0; dataPtn < population.rSize(); dataPtn++)
        {
            if(timeIsFirstCol == true && col == 0)
            {
                stdPop(dataPtn, col) = population(dataPtn, col);
            }
            else
            {
                stdPop(dataPtn, col) =  population(dataPtn, col) / stdDeviations[nonTimeDataIndex];
            }
        }
        Log(lDebug)<<"Standard deviation: "<<stdDeviations[nonTimeDataIndex];
    }

    return stdPop;
}

vector<double> getStandardDeviations(const TelluriumData& population)
{
    vector<double> means = getMeans(population);

    //check for time column
    bool timeIsFirstCol = population.isFirstColumnTime();

    vector<double> stds;
    for(int col = (timeIsFirstCol == true) ? 1 : 0; col < population.cSize(); col++)
    {
        double sumOfSquaredDifferences  = 0;
        for(int row = 0; row < population.rSize(); row++)
        {
            int meansIndex = (col - (timeIsFirstCol == true)) ? 1 : 0;
            sumOfSquaredDifferences += pow( population(row, col) - means[meansIndex], 2);
        }
        double variance =  (1. /  (population.rSize() -1) ) * sumOfSquaredDifferences ;
        double stdDev =  sqrt(variance) ;
        Log(lInfo) << "Std Dev = " << stdDev;
        stds.push_back(stdDev);
    }
    return stds;
}

double getStandardDeviation(const vector<double>& population, double* theMean)
{
    double mean = getMean(population);
    if(theMean)
    {
        (*theMean) = mean;
    }

    double sumOfSquaredDifferences  = 0;
    for(int row = 0; row < population.size(); row++)
    {
        sumOfSquaredDifferences += pow( population[row] - mean, 2);
    }
    double variance =  (1. /  (population.size() -1) ) * sumOfSquaredDifferences;
    double stdDev =  sqrt(variance) ;
    Log(lInfo) << "Std Dev = " << stdDev;

    return stdDev;
}

double getMean(const vector<double>& population)
{
    if(population.size() < 1)
    {
        throw(Exception("Bad population passed to function getMean()"));
    }

    double mean  = 0;
    for(int row = 0; row < population.size(); row++)
    {
        mean += population[row];

    }
    return mean/population.size();
}

vector<double> getMeans(const TelluriumData& population)
{
    if(population.rSize() < 1)
    {
        throw(Exception("Bad population passed to function getMeans()"));
    }

    vector<double> means;
    //check for time column
    bool timeIsFirstCol = population.isFirstColumnTime();

    for(int col = (timeIsFirstCol == true) ? 1 : 0; col < population.cSize(); col++)
    {
        double mean  = 0;
        for(int row = 0; row < population.rSize(); row++)
        {
            mean += population(row, col);

        }
        means.push_back(mean / population.rSize());
    }
    return means;
}

double erf(double x)
{
    double pi = 3.1415927;
    double a = (8.*(pi - 3.))/(3.*pi*(4. - pi));
    double x2 = x * x;

    double ax2 = a * x2;
    double num = (4./pi) + ax2;
    double denom = 1.0 + ax2;

    double inner = (-1. * x2)* num/ denom;
    double erf2 = 1.0 - exp(inner);
    double val = sqrt(erf2);
    if(x >= 0)
    {
        return  val;
    }
    else
    {
        return -1.0 * val;
    }
}

double cdf(double n)
{
    if(n < 0)
    {
            return (1. - erf(n / sqrt(2.)))/2.;
    }
    else
    {
            return (1. + erf(n / sqrt(2.)))/2.;
    }
}

}

