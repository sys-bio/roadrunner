#include <iostream>
#include <iomanip>
#include "rrc_api.h"

using namespace std;
using namespace rrc;

int main()
{
    char* modelFile("../models/test_1.xml");
    RRHandle rrHandle = createRRInstance();

    if (!loadSBML(rrHandle, modelFile))
    {
       cout << "Error while loading SBML file\n";
       cout << "Error message: "<< getLastError() <<endl;
       exit(-1);
    }

    simulateEx (rrHandle, 0, 10, 10);  // start time, end time, and number of points
    RRCData* cOutput = getSimulationResult(rrHandle);

    int index = 0;
    for(int col = 0; col < cOutput->CSize; col++)
    {
        cout<<setw(20)<<left<<cOutput->ColumnHeaders[index++];
        if(col < cOutput->CSize -1)
        {
            cout<<"\t";
        }
    }
    cout<<"\n";
    index =0;
    for(int row = 0; row < cOutput->RSize; row++)
    {
        for(int col = 0; col < cOutput->CSize; col++)
        {
            cout<<setw(20)<<left<<cOutput->Data[index++];
            if(col < cOutput->CSize -1)
            {
                cout<<"\t";
            }
        }
        cout<<"\n";

      }

    freeRRCData (cOutput);
    freeRRInstance (rrHandle);
    return 0;
}


