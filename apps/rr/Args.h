#ifndef CommandLineParametersH
#define CommandLineParametersH
#include <string>
#include "rrLogger.h"

using std::string;

string Usage(const string& prg);
class Args
{
public:
    Args();
    virtual                        ~Args(){}
    rr::Logger::Level               CurrentLogLevel;    //option v:
    string                          ModelFileName;      //option m:
    string                          OutputFileName;     //option o
    string                          DataOutputFolder;   //option d:
    string                          TempDataFolder;     //option t:
    bool                            Pause;              //option p
    std::string                     compilerStr;        //option c
    bool                            UseOSTempFolder;    //option u
    double                          StartTime;          //option s
    double                          Duration;
    double                          EndTime;            //option e
    int                             Steps;              //option z
    string                          SelectionList;      //option l:
    bool variableStep;
};

#endif
