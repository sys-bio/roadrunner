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

void rrcCallConv setModelName(RRHandle handle, char* name)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        rri->setModelName(name);
    catch_void_macro
}

char* rrcCallConv getModelId(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
    return createText(rri->getModelId());
    catch_ptr_macro
}

void rrcCallConv setModelId(RRHandle handle, char* id)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
    rri->setModelId(id);
    catch_void_macro
}

int rrcCallConv getNumberOfRules(RRHandle handle)
{
    start_try
        RoadRunner* rri = castToRoadRunner(handle);
        if(!rri->getModel())
        {
            rrLog(Logger::LOG_WARNING)<<"Model is not allocated.";
            return -1;
        }
        int value = rri->getModel()->getNumRateRules();
        return value;
    catch_int_macro
}

} //namespace rrc
