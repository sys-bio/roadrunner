#pragma hdrstop
#include "rrLogger.h"
#include "telplugins_logging_api.h"
#include "telplugins_cpp_support.h"
#include "rrRoadRunner.h"
//---------------------------------------------------------------------------

using namespace std;
using namespace rr;
using namespace tlpc;

bool tlp_cc tpEnableLoggingToConsole()
{
    start_try
        Logger::enableConsoleLogging();
        return true;
    catch_bool_macro
}

bool tlp_cc tpDisableLoggingToConsole()
{
    start_try
        Logger::disableConsoleLogging();
        return true;
    catch_bool_macro
}

bool tlp_cc tpEnableLoggingToFile(char* fName)
{
    start_try
//        char* tempFolder = getTempFolder(handle);
        string logFile = string(fName);//joinPath(tempFolder, "RoadRunner.log") ;
//        tel::freeText(tempFolder);

        Logger::enableFileLogging(logFile);
        return true;
    catch_bool_macro
}

bool tlp_cc tpDisableLoggingToFile()
{
    start_try
        Logger::disableFileLogging();
        return true;
    catch_bool_macro
}

bool tlp_cc tpSetLogLevel(const char* _lvl)
{
    start_try
        Logger::Level lvl = Logger::stringToLevel(_lvl);
        rr::Logger::setLevel(lvl);
        return true;
    catch_bool_macro
}

char* tlp_cc tpGetLogLevel()
{
    start_try
        string level = Logger::getCurrentLevelAsString();
        char* lvl = tpCreateText(level.c_str());
        return lvl;
    catch_ptr_macro
}

char* tlp_cc tpGetLogFileName()
{
    start_try
        return tpCreateText(Logger::getFileName().c_str());
    catch_ptr_macro
}

void tlp_cc tpLogMsg(CLogLevel lvl, const char* msg)
{
    start_try
        if(msg)
        {
            Log((LogLevel) lvl)<<msg;
        }
        else
        {
            Log((LogLevel) lvl)<<"You passed a NULL message pointer to logMsg";
        }
    catch_void_macro
}
