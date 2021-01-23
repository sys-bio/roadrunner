#ifndef lmUtilsH
#define lmUtilsH
#include "telplugins_types.h"
#include "telplugins_settings.h"
#include "telPluginsManagerSettings.h"
#include "rrRoadRunner.h"
//---------------------------------------------------------------------------

namespace lmfit
{
//using namespace rrc;
using namespace tlpc;

/* data structure to transmit model data to function evalution */
typedef struct
{
    int                     nrOfSpecies;
    int                     nrOfParameters;                 //Number of parameters to fit
    int                     nrOfTimePoints;
    int                     nrOfResiduePoints;              //This is the number "m_dat", typically nrOfSpecies * nrOfTimePoints
    double*                 parameters;                     //Vector holding the numbers for each parameter
    char**                  speciesLabels;
    char**                  parameterLabels;                //Labels for the parameters to fit, eg k1, k2 etc
    double**                experimentalData;               //Species experimental time data
    double**                experimentalDataWeights;        //experimental weights..
    double                  timeStart;
    double                  timeEnd;
    double*                 time;
    TELHandle               mLMPlugin;
    PluginEvent             mProgressEvent;
    void*                   mProgressEventContextData;
    char*                   mProgressEventMessage;
    int                     mNrOfIterations;
    double                  mNorm;
    rr::RoadRunner*         roadrunner;
} lmDataStructure;

//Evaluate is the event function that lmfit call to evaulate residuals
void evaluate(      const double   *par,                    //Parameter vector
                    int             m_dat,                  //Dimension of residue vector
                    const void     *userData,               //Data structure
                    double         *fvec,                   //residue vector..
                    int            *userBreak               //Index into info message array
                 );
}
#endif
