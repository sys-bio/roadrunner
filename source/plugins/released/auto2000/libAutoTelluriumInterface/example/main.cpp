#pragma hdrstop
#include "rrLogger.h"
#include "rrUtils.h"
#include "rrRoadRunner.h"
#include "rrException.h"
#include "../rrRRAuto.h"


using namespace rrauto;
int main(int argc, char* argv[])
{
    try
    {
        gLog.setLevel(lInfo);
        string tempFolder("r:/temp");
        RoadRunner rr;
        AutoData data;
        RRAuto rrAuto(&rr, data);

        rr.setTempFileFolder(tempFolder);
        string testFile("../models/bistable.xml");
        if(!rr.load(testFile))
        {
            throw(rr::Exception("Failed loading SBML"));
        }

        rrAuto.selectParameter("k3");
        rrAuto.setStartParameterValue(0.2);
        rrAuto.setEndParameterValue(1.5);
        rrAuto.setScanDirection(sdNegative);

        Log(lInfo)<<rrAuto.getConstantsAsString();

        rrAuto.run();

        //Print the result
        Log(lInfo)<<rrAuto.getAutoData();
        pause(true);
    }
    catch(const rr::Exception& ex)
    {
        Log(lError)<<"There was a problem: "<<ex.getMessage();
    }
    catch (std::exception &e)
    {
        Log(lError) << "caught Exception: " << e.what() << "\n";
    }
    return 0;
}


