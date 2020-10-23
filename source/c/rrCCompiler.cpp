#pragma hdrstop
#include <sstream>
#if defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#include <strsafe.h>
#if defined(__CODEGEARC__)
    #include <dir.h>
#elif defined(_MSVC)
    #include <direct.h>
#endif
#endif
#include "Poco/File.h"
#include "rrLogger.h"
#include "rrCCompiler.h"
#include "rrException.h"
#include "rrStringUtils.h"
#include "rrUtils.h"
#include "rrRoadRunner.h"
//---------------------------------------------------------------------------

using namespace std;
namespace rr
{

CCompiler::CCompiler(const string& supportCodeFolder, const string& compiler)
:
mSupportCodeFolder(supportCodeFolder),
mCompilerName(getFileName(compiler)),
mCompilerLocation(getFilePath(compiler))
{
    if(mSupportCodeFolder.size() > 0)
    {
        if(!setupCompiler(mSupportCodeFolder))
        {
            Log(lWarning)<<"Roadrunner internal compiler setup failed. ";
        }
    }
}

CCompiler::~CCompiler(){}

bool CCompiler::setupCompiler(const string& supportCodeFolder)
{
    mSupportCodeFolder = supportCodeFolder;

    if(!folderExists(mSupportCodeFolder))
    {
        Log(Logger::LOG_ERROR)<<"The roadrunner support code folder : "<<mSupportCodeFolder<<" does not exist.";
        return false;
    }

    return true;
}

bool CCompiler::setOutputPath(const string& path)
{
    mOutputPath = path;
    return true;
}

bool CCompiler::compileSource(const string& sourceFileName)
{
    //Compile the code and load the resulting dll, and call an exported function in it...
#if defined(_WIN32) || defined(__CODEGEARC__)
    string dllFName(changeFileExtensionTo(getFileName(sourceFileName), "dll"));
#elif defined(__unix__)
    string dllFName(changeFileExtensionTo(getFileName(sourceFileName), "so"));
#elif defined(__APPLE__)
    string dllFName(changeFileExtensionTo(getFileName(sourceFileName), "dylib"));
#endif
    mDLLFileName = joinPath(getFilePath(sourceFileName), dllFName);

    //Setup compiler environment
    setupCompilerEnvironment();

    string exeCmd = createCompilerCommand(sourceFileName);

    //exeCmd += " > compileLog.log";
    Log(lDebug2)<<"Compiling model..";
    Log(lDebug)<<"\nExecuting compile command: "<<exeCmd;

    if(!compile(exeCmd))
    {
        Log(Logger::LOG_ERROR)<<"Creating DLL failed..";
        throw Exception("Creating Model DLL failed..");
    }

    //Check if the DLL exists...
    return fileExists(mDLLFileName);
}

bool CCompiler::setCompiler(const string& compiler)
{
    mCompilerName = getFileName(compiler);
    mCompilerLocation = getFilePath(compiler);
    return true;
}

string CCompiler::getCompiler() const
{
    return mCompilerName;
}

bool CCompiler::setCompilerLocation(const string& path)
{
    if(!folderExists(path))
    {
        Log(Logger::LOG_ERROR)<<"Tried to set invalid path: "<<path<<" for compiler location";
        return false;
    }
    mCompilerLocation = path;
    return true;
}

string CCompiler::getCompilerLocation() const
{
    return mCompilerLocation;
}

bool CCompiler::setSupportCodeFolder(const string& path)
{
    if(!folderExists(path))
    {
        Log(Logger::LOG_ERROR)<<"Tried to set invalid path: "<<path<<" for compiler location";
        return false;
    }
    mSupportCodeFolder = path;
    return true;
}

string CCompiler::getSupportCodeFolder() const
{
    return mSupportCodeFolder;
}

bool CCompiler::setupCompilerEnvironment()
{
    mIncludePaths.clear();
    mLibraryPaths.clear();
    mCompilerFlags.clear();
    if(getFileNameNoExtension(mCompilerName) == "tcc" || getFileNameNoExtension(mCompilerName) == "gcc")
    {
        mCompilerFlags.push_back("-g");         //-g adds runtime debug information
#if defined(__unix__) || defined(_WIN32)
        mCompilerFlags.push_back("-shared");
        mCompilerFlags.push_back("-rdynamic");  //-rdynamic : Export global symbols to the dynamic linker
#elif defined(__APPLE__)
        mCompilerFlags.push_back("-dynamiclib");
#endif
                                                //-b : Generate additional support code to check memory allocations and array/pointer bounds. `-g' is implied.

        mCompilerFlags.push_back("-fPIC"); // shared lib
        mCompilerFlags.push_back("-O0"); // turn off optimization

        //LogLevel                              //-v is for verbose
        if(getFileNameNoExtension(mCompilerName) == "tcc")
        {
            mIncludePaths.push_back(".");
            mIncludePaths.push_back("r:/rrl/source");

            mIncludePaths.push_back(joinPath(mCompilerLocation, "include"));
            mLibraryPaths.push_back(".");
            mLibraryPaths.push_back(joinPath(mCompilerLocation, "lib"));
            if(gLog.getLevel() < lDebug)
            {
                mCompilerFlags.push_back("-v"); // suppress warnings
            }
            else if(gLog.getLevel() >= lDebug1)
            {
                mCompilerFlags.push_back("-vv");
            }
            else if(gLog.getLevel() >= lDebug2)
            {
                mCompilerFlags.push_back("-vvv");
            }
        }
        else if(getFileNameNoExtension(mCompilerName) == "gcc")
        {
            if(gLog.getLevel() < lDebug)
            {
                mCompilerFlags.push_back("-w"); // suppress warnings
            }
            else if(gLog.getLevel() >= lDebug1)
            {
                mCompilerFlags.push_back("-Wall");
            }
            else if(gLog.getLevel() >= lDebug2)
            {
                mCompilerFlags.push_back("-Wall -pedantic");
            }
        }
    }

    mIncludePaths.push_back(mSupportCodeFolder);
    return true;
}

string CCompiler::createCompilerCommand(const string& sourceFileName)
{
    stringstream exeCmd;
    if(getFileNameNoExtension(mCompilerName) == "tcc"
       || getFileNameNoExtension(mCompilerName) == "gcc"
       || getFileNameNoExtension(mCompilerName) == "cc")
    {
        // standard unix compiler options
        exeCmd<<joinPath(mCompilerLocation, mCompilerName);
        //Add compiler flags
        for(int i = 0; i < mCompilerFlags.size(); i++)
        {
            exeCmd<<" "<<mCompilerFlags[i];
        }
        exeCmd<<" \""<<sourceFileName<<"\" \""<<joinPath(mSupportCodeFolder, "rrSupport.c")<<"\"";


        exeCmd<<" -o \""<<mDLLFileName<<"\"";
#if defined(WIN32)
        exeCmd<<" -DBUILD_MODEL_DLL ";
#endif
        //Add include paths
        for(int i = 0; i < mIncludePaths.size(); i++)
        {
            exeCmd<<" -I\""<<mIncludePaths[i]<<"\" " ;
        }

        //Add library paths
        for(int i = 0; i < mLibraryPaths.size(); i++)
        {
            exeCmd<<" -L\""<<mLibraryPaths[i]<<"\" " ;
        }
    }
    return exeCmd.str();
}

#ifdef WIN32

bool CCompiler::compile(const string& cmdLine)
{
    if( !cmdLine.size() )
    {
        return false;
    }

    PROCESS_INFORMATION pi;
    ZeroMemory( &pi, sizeof(pi) );

    STARTUPINFO si;
    ZeroMemory( &si, sizeof(si) );
    si.cb = sizeof(si);

    //sec attributes for the output file
    SECURITY_ATTRIBUTES sao;
    sao.nLength=sizeof(SECURITY_ATTRIBUTES);
    sao.lpSecurityDescriptor=NULL;
    sao.bInheritHandle=1;

    string compilerTempFile(joinPath(mOutputPath, getFileNameNoExtension(mDLLFileName)));
    compilerTempFile.append("C.log");

    Poco::File aFile(compilerTempFile);
    if(aFile.exists())
    {
        aFile.remove();
    }

    HANDLE outFile;
      //Todo: there is a problem creating the logfile after first time creation..
    if((outFile=CreateFileA(compilerTempFile.c_str(),
                            GENERIC_WRITE,
                            FILE_SHARE_DELETE,
                            &sao,
                            OPEN_ALWAYS,
                            FILE_ATTRIBUTE_NORMAL,
                            NULL))==INVALID_HANDLE_VALUE)
    {
        // Retrieve the system error message for the last-error code
        DWORD errorCode = GetLastError();
        string anError = getWINAPIError(errorCode, TEXT("CreateFile"));
        Log(Logger::LOG_ERROR)<<"WIN wrappers Error (after CreateFile): "<<anError;
        Log(Logger::LOG_ERROR)<<"Failed creating logFile for compiler output";
    }

    SetFilePointer(outFile, 0, NULL, FILE_END); //set pointer position to end file

    //init the STARTUPINFO struct
    si.dwFlags=STARTF_USESTDHANDLES;
    si.hStdOutput = outFile;
    si.hStdError  = outFile;

    //proc sec attributes
    SECURITY_ATTRIBUTES sap;
    sap.nLength=sizeof(SECURITY_ATTRIBUTES);
    sap.lpSecurityDescriptor=NULL;
    sap.bInheritHandle=1;

    //thread sec attributes
    SECURITY_ATTRIBUTES sat;
    sat.nLength=sizeof(SECURITY_ATTRIBUTES);
    sat.lpSecurityDescriptor=NULL;
    sat.bInheritHandle=1;

    // Start the child process.
    if( !CreateProcessA(
        NULL,                           // No module name (use command line)
        (char*) cmdLine.c_str(),        // Command line
        &sap,                           // Process handle not inheritable
        &sat,                           // Thread handle not inheritable
        TRUE,                          // Set handle inheritance
        CREATE_NO_WINDOW,               // Creation flags
        NULL,                           // Use parent's environment block
        NULL,                           // Use parent's starting directory
        &si,                            // Pointer to STARTUPINFO structure
        &pi )                           // Pointer to PROCESS_INFORMATION structure
    )
    {
        DWORD errorCode = GetLastError();

        string anError = getWINAPIError(errorCode, TEXT("CreateProcess"));
        Log(Logger::LOG_ERROR)<<"WIN wrappers Error: (after CreateProcess) "<<anError;

        // Close process and thread handles.
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
        CloseHandle(outFile);
        return false;
    }

    // Wait until child process exits.
    WaitForSingleObject(pi.hProcess, INFINITE);

    CloseHandle(outFile);

    // Close process and thread handles.
    CloseHandle(pi.hProcess);
    DWORD errorCode = GetLastError();
    if(errorCode != 0)
    {
        string anError = getWINAPIError(errorCode, TEXT("CloseHandle"));
        Log(lDebug)<<"WIN wrappers error: (pi.hProcess)"<<anError;
    }

    CloseHandle(pi.hThread);
    errorCode = GetLastError();
    if(errorCode != 0)
    {
        string anError = getWINAPIError(errorCode, TEXT("CloseHandle"));
        Log(lDebug)<<"WIN wrappers error: (pi.hThread)"<<anError;
    }

    //Read the log file and log it
    if(fileExists(compilerTempFile))
    {
        string log = getFileContent(compilerTempFile.c_str());
        Log(lDebug)<<"Compiler output: "<<log<<endl;
    }

    return true;
}

#else  //---------------- LINUX, UNIXES

bool CCompiler::compile(const string& cmdLine)
{
    string toFile(cmdLine);
    toFile += " >> ";
    toFile += joinPath(mOutputPath, "compilation.log");
    toFile += " 2>&1";

    Log(lDebug)<<"Compiler command: "<<toFile;

    //Create the shared library, using system call
    int val = system(toFile.c_str());
    if(val == 0)
    {
        Log(lDebug)<<"Compile system call was succesful";
        return true;
    }
    else
    {
        Log(Logger::LOG_ERROR)<<"Compile system call returned: "<<val;
        return false;
    }
}

#endif //WIN32

std::string CCompiler::getDefaultTargetTriple()
{
    throw rr::Exception(std::string(__FUNC__) + " not supported with legacy C back end");
}

std::string CCompiler::getProcessTriple()
{
    throw rr::Exception(std::string(__FUNC__) + " not supported with legacy C back end");
}

std::string CCompiler::getHostCPUName()
{
    throw rr::Exception(std::string(__FUNC__) + " not supported with legacy C back end");
}

std::string CCompiler::getVersion()
{
    throw rr::Exception(std::string(__FUNC__) + " not supported with legacy C back end");
}

string getCompilerMessages()
{
    return "No messages yet";
}

} //namespace rr

