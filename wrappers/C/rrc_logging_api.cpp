#pragma hdrstop
#include "rrLogger.h"
#include "rrException.h"
#include "rrc_api.h"
#include "rrc_logging_api.h"
#include "rrc_cpp_support.h"
#include "rrStringUtils.h"
//---------------------------------------------------------------------------

namespace rrc
{
using namespace rr;
using namespace std;

bool rrcCallConv enableLoggingToConsole()
{
    start_try
        Logger::enableConsoleLogging();
        return true;
    catch_bool_macro
}

bool rrcCallConv disableLoggingToConsole()
{
    start_try
        Logger::disableConsoleLogging();
        return true;
    catch_bool_macro
}

bool rrcCallConv enableLoggingToFile(RRHandle handle)
{
    start_try
        char* tempFolder = getTempFolder(handle);
        string logFile = joinPath(tempFolder, "RoadRunner.log") ;
        rr::freeText(tempFolder);

        Logger::enableFileLogging(logFile);
        return true;
    catch_bool_macro
}

bool rrcCallConv disableLoggingToFile()
{
    start_try
        Logger::disableFileLogging();
        return true;
    catch_bool_macro
}

bool rrcCallConv setLogLevel(const char* _lvl)
{
    start_try
        Logger::Level lvl = Logger::stringToLevel(_lvl);
        Logger::setLevel(lvl);
        return true;
    catch_bool_macro
}

char* rrcCallConv getLogLevel()
{
    start_try
        string level = Logger::getCurrentLevelAsString();
        char* lvl = createText(level.c_str());
        return lvl;
    catch_ptr_macro
}

char* rrcCallConv getLogFileName()
{
    start_try
        return createText(Logger::getFileName().c_str());
    catch_ptr_macro
}

void rrcCallConv logMsg(CLogLevel lvl, const char* msg)
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

}
