#pragma hdrstop
#include "rrRoadRunner.h"
#include "rrExecutableModel.h"
#include "rrException.h"
#include "rrc_nom_api.h"
#include "rrc_cpp_support.h"
#include "rrc_macros.h"
#include "rrStringUtils.h"
#include "rrLogger.h"
//---------------------------------------------------------------------------
namespace rrc
{
using namespace rr;
using namespace std;
//NOM forwarded functions

char* rrcCallConv getModelName(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        return createText(rri->getModelName());
    catch_ptr_macro
}

int rrcCallConv getNumberOfRules(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        if(!rri->getModel())
        {
            Log(Logger::LOG_WARNING)<<"Model is not allocated.";
            return -1;
        }
        int value = rri->getModel()->getNumRateRules();
        return value;
    catch_int_macro
}

} //namespace rrc
