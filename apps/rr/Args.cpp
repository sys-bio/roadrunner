#pragma hdrstop
#include <iomanip>
#include <sstream>
#include "Args.h"
#include "rrVersionInfo.h"
#include "rrConfig.h"

using namespace std;

Args::Args()
:
UseOSTempFolder(false),
compilerStr(""),
Pause(false),
CurrentLogLevel(rr::Logger::LOG_WARNING),
ModelFileName(""),
DataOutputFolder(""),
TempDataFolder("."),
StartTime(0),
Duration(5),
EndTime(StartTime + Duration),
Steps(50),
SelectionList(""),
variableStep(false)
{}

string Usage(const string& prg)
{
    Args args;
    stringstream usage;
    usage<< "rr.exe is a command line driven SBML simulator\n\n";
    usage << "\nUSAGE for "<<prg<<"\n\n";
    usage<<left;
    usage<<setfill('.');
    usage<<setw(25)<<"-v<debug level>"              <<" Debug levels: Error, Warning, Info, Debug. Default: Info\n";
    usage<<setw(25)<<"-m<FileName>"                 <<" SBML Model File Name (with path)\n";
    usage<<setw(25)<<"-o<FileName>"                 <<" FileName for data output \n";
    usage<<setw(25)<<"-d<FilePath>"                 <<" Data output directory. If not given, data is output to current directory (implies -f is given)\n";
    usage<<setw(25)<<"-t<FilePath>"                 <<" Temporary data output directory. If not given, temp files are output to current directory\n";
    usage<<setw(25)<<"-p"                           <<" Pause before exiting.\n";
    usage<<setw(25)<<"-c"                           <<" Compiler string, to enable MCJIT, this should contain \"use_mcjit\" \n";
    usage<<setw(25)<<"-u"                           <<" Use users OS designated temporary directory\n";
    usage<<setw(25)<<"-s<#>"                        <<" Set the start time for simulation. Default: 0\n";
    usage<<setw(25)<<"-e<#>"                        <<" Set the end time for simulation. Default: 5\n";
    usage<<setw(25)<<"-z<#>"                        <<" Set number of steps in the simulation. Default: 50\n";
    usage<<setw(25)<<"-l<List>"                     <<" Set selection list. Separate variables using ',' or space\n";
    usage<<setw(25)<<"-version"                     <<" Prints the current version.\n\n";
    usage<<setw(25)<<"-x"                           <<" Enables variable step mode.\n\n";
    usage<<setw(25)<<"-? "                          <<" Shows the help screen.\n\n";

    usage << "Version: " << rr::getVersionStr() << std::endl;
    usage << "config file: " << rr::Config::getConfigFilePath() << endl;
    return usage.str();
}

