#ifndef lmUtilsH
#define lmUtilsH
#include "telplugins_types.h"
#include "telplugins_settings.h"
#include "telPluginsManagerSettings.h"
#include <string>
#include <vector>
//---------------------------------------------------------------------------

namespace lmfit
{
using namespace tlpc;
class LM;

/* data structure to transmit model data to function evalution */
struct lmDataStructure
{
    int                     nrOfSpecies;
    int                     nrOfParameters;                 //Number of parameters to fit
    int                     nrOfTimePoints;
    int                     nrOfResiduePoints;              //This is the number "m_dat", typically nrOfSpecies * nrOfTimePoints
    std::vector<double>                 parameters;                 //Vector holding the numbers for each parameter
    std::vector<std::string>            speciesLabels;
    std::vector<std::string>            parameterLabels;            //Labels for the parameters to fit, eg k1, k2 etc
    std::vector<std::vector<double>>    experimentalData;           //Species experimental time data
    std::vector<std::vector<double>>    experimentalDataWeights;    //experimental weights..
    double                  timeStart;
    double                  timeEnd;
    std::vector<double>     time;
    TELHandle               mLMPlugin;
    PluginEvent             mProgressEvent;
    void*                   mProgressEventContextData;
    char*                   mProgressEventMessage;
    int                     mNrOfIterations;
    double                  mNorm;
    LM*                     mTheHost;
};

//Evaluate is the event function that lmfit call to evaulate residuals
//void evaluate(      const double   *par,                    //Parameter vector
//                    int             m_dat,                  //Dimension of residue vector
//                    const void     *userData,               //Data structure
//                    double         *fvec,                   //residue vector..
//                    int            *userBreak               //Index into info message array
//                 );
}
#endif
