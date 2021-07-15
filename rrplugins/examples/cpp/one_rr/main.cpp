#include <iostream>

#include "rrRoadRunner.h"
#include "rrRoadRunnerData.h"
#include "rrUtils.h"
#include "C/rrc_api.h"
#include "telLogger.h"
#include "telException.h"
#include "telplugins_c_api.h"

using namespace rr;
using namespace rrc;
using namespace tlp;
using namespace std;

int main(int argc, char** argv)
{
    const char* rootPath = "..";

    try
    {
        tlp::Logger::setLevel(tlp::Logger::LOG_DEBUG);
        tlp::Logger::enableFileLogging("Test.log");
        string model = "bistable.xml";
        const string modelFile = joinPath(rootPath, "models", model);

        RoadRunner rr1;
        LoadSBMLOptions opt;

        //opt.modelGeneratorOpt |= LoadSBMLOptions::RECOMPILE;

        rr1.load(modelFile, &opt);

        SimulateOptions simOpt;
        simOpt.start = 0;
        simOpt.start = 10;
        simOpt.steps= 511;

        rr1.setGlobalParameterByIndex(2, 0.2);
        double ss = rr1.steadyState();

        RRPLOG(tlp::lInfo) << "Steady State: " <<ss;
        const ls::DoubleMatrix* rrData = rr1.simulate(&simOpt);
        //simulate((RRHandle) &rr1)

        //TELHandle dataHandle = tpCreateTelluriumDataFromRoadRunnerData( (TELHandle) rrData);

        unsigned int nrRows = rrData->numRows();
        unsigned int nrCols = rrData->numCols();
        const vector<string>& colHeaders = rrData->getColNames();

        cout<<"NrRows: "<<nrRows<<endl;
        cout<<"NrCols "<<nrCols<<endl;
        cout << "Headers: ";
        for (size_t h = 0; h < colHeaders.size(); h++) {
            cout << colHeaders[h] << " ";
        }
        cout << endl;
        for(int row = 0; row < nrRows; row++)
        {
            cout << "row: " << row << "\t";
            for(int col = 0; col < nrCols; col++)
            {
                cout << rrData[row][col] << "\t";
            }
            cout << endl;
        }

    }
    catch(const exception& ex)
    {

        RRPLOG(tlp::Logger::LOG_ERROR)<<"There was a problem: "<<ex.what();
    }

    return 0;
}




