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


namespace rr
{

CCompiler::CCompiler(const std::string& supportCodeFolder, const std::string& compiler)
:
mSupportCodeFolder(supportCodeFolder),
mCompilerName(getFileName(compiler)),
mCompilerLocation(getFilePath(compiler))
{
    if(mSupportCodeFolder.size() > 0)
    {
        if(!setupCompiler(mSupportCodeFolder))
        {
            rrLog(lWarning)<<"Roadrunner internal compiler setup failed. ";
        }
    }
}

CCompiler::~CCompiler(){}

bool CCompiler::setupCompiler(const std::string& supportCodeFolder)
{
    mSupportCodeFolder = supportCodeFolder;

    if(!std::filesystem::is_directory(mSupportCodeFolder))
    {
        rrLog(Logger::LOG_ERROR)<<"The roadrunner support code folder : "<<mSupportCodeFolder<<" does not exist.";
        return false;
    }

    return true;
}

bool CCompiler::setOutputPath(const std::string& path)
{
    mOutputPath = path;
    return true;
}

bool CCompiler::compileSource(const std::string& sourceFileName)
{
    //Compile the code and load the resulting dll, and call an exported function in it...
#if defined(_WIN32) || defined(__CODEGEARC__)
    std::string dllFName(changeFileExtensionTo(getFileName(sourceFileName), "dll"));
#elif defined(__unix__)
    std::string dllFName(changeFileExtensionTo(getFileName(sourceFileName), "so"));
#elif defined(__APPLE__)
    std::string dllFName(changeFileExtensionTo(getFileName(sourceFileName), "dylib"));
#endif
    mDLLFileName = (std::filesystem::path(getFilePath(sourceFileName)) / dllFName).string();

    //Setup compiler environment
    setupCompilerEnvironment();

    std::string exeCmd = createCompilerCommand(sourceFileName);

    //exeCmd += " > compileLog.log";
    rrLog(lDebug2)<<"Compiling model..";
    rrLog(lDebug)<<"\nExecuting compile command: "<<exeCmd;

    if(!compile(exeCmd))
    {
        rrLog(Logger::LOG_ERROR)<<"Creating DLL failed..";
        throw Exception("Creating Model DLL failed..");
    }

    //Check if the DLL exists...
    return std::filesystem::exists(mDLLFileName);
}

bool CCompiler::setCompiler(const std::string& compiler)
{
    mCompilerName = getFileName(compiler);
    mCompilerLocation = getFilePath(compiler);
    return true;
}

std::string CCompiler::getCompiler() const
{
    return mCompilerName;
}

bool CCompiler::setCompilerLocation(const std::string& path)
{
    if(!std::filesystem::is_directory(path))
    {
        rrLog(Logger::LOG_ERROR)<<"Tried to set invalid path: "<<path<<" for compiler location";
        return false;
    }
    mCompilerLocation = path;
    return true;
}

std::string CCompiler::getCompilerLocation() const
{
    return mCompilerLocation;
}

bool CCompiler::setSupportCodeFolder(const std::string& path)
{
    if(!std::filesystem::is_directory(path))
    {
        rrLog(Logger::LOG_ERROR)<<"Tried to set invalid path: "<<path<<" for compiler location";
        return false;
    }
    mSupportCodeFolder = path;
    return true;
}

std::string CCompiler::getSupportCodeFolder() const
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

            mIncludePaths.push_back((std::filesystem::path(mCompilerLocation) / "include").string());
            mLibraryPaths.push_back(".");
            mLibraryPaths.push_back((std::filesystem::path(mCompilerLocation) / "lib").string());
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

std::string CCompiler::createCompilerCommand(const std::string& sourceFileName)
{
    std::stringstream exeCmd;
    if(getFileNameNoExtension(mCompilerName) == "tcc"
       || getFileNameNoExtension(mCompilerName) == "gcc"
       || getFileNameNoExtension(mCompilerName) == "cc")
    {
        // standard unix compiler options
        exeCmd<<(std::filesystem::path(mCompilerLocation) / mCompilerName).string();
        //Add compiler flags
        for(int i = 0; i < mCompilerFlags.size(); i++)
        {
            exeCmd<<" "<<mCompilerFlags[i];
        }
        exeCmd<<" \""<<sourceFileName<<"\" \""<<(std::filesystem::path(mSupportCodeFolder) / "rrSupport.c").string()<<"\"";


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

bool CCompiler::compile(const std::string& cmdLine)
{
    if( !cmdLine.size() )
    {
        return false;
    }

    PROCESS_INFORMATION pi;
    ZeroMemory( &pi, sizeof(pi) );

    STARTUPINFOA si;
    ZeroMemory( &si, sizeof(si) );
    si.cb = sizeof(si);

    //sec attributes for the output file
    SECURITY_ATTRIBUTES sao;
    sao.nLength=sizeof(SECURITY_ATTRIBUTES);
    sao.lpSecurityDescriptor=NULL;
    sao.bInheritHandle=1;

    std::string compilerTempFile((std::filesystem::path(mOutputPath) / getFileNameNoExtension(mDLLFileName)).string());
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
        TCHAR funcNameBuf[] = TEXT("CreateFile");
        std::string anError = getWINAPIError(errorCode, funcNameBuf);
        rrLog(Logger::LOG_ERROR)<<"WIN wrappers Error (after CreateFile): "<<anError;
        rrLog(Logger::LOG_ERROR)<<"Failed creating logFile for compiler output";
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

        TCHAR funcNameBuf[] = TEXT("CreateProcess");
        std::string anError = getWINAPIError(errorCode, funcNameBuf);
        rrLog(Logger::LOG_ERROR)<<"WIN wrappers Error: (after CreateProcess) "<<anError;

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
        TCHAR funcNameBuf[] = TEXT("CloseHandle");
        std::string anError = getWINAPIError(errorCode, funcNameBuf);
        rrLog(lDebug)<<"WIN wrappers error: (pi.hProcess)"<<anError;
    }

    CloseHandle(pi.hThread);
    errorCode = GetLastError();
    if(errorCode != 0)
    {
        TCHAR funcNameBuf[] = TEXT("CloseHandle");
        std::string anError = getWINAPIError(errorCode, funcNameBuf);
        rrLog(lDebug)<<"WIN wrappers error: (pi.hThread)"<<anError;
    }

    //Read the log file and log it
    if(std::filesystem::exists(compilerTempFile))
    {
        std::string log = getFileContent(compilerTempFile.c_str());
        rrLog(lDebug)<<"Compiler output: "<<log<<std::endl;
    }

    return true;
}

#else  //---------------- LINUX, UNIXES

bool CCompiler::compile(const std::string& cmdLine)
{
    std::string toFile(cmdLine);
    toFile += " >> ";
    toFile += (std::filesystem::path(mOutputPath) / "compilation.log").string();
    toFile += " 2>&1";

    rrLog(lDebug)<<"Compiler command: "<<toFile;

    //Create the shared library, using system call
    int val = system(toFile.c_str());
    if(val == 0)
    {
        rrLog(lDebug)<<"Compile system call was succesful";
        return true;
    }
    else
    {
        rrLog(Logger::LOG_ERROR)<<"Compile system call returned: "<<val;
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

std::string getCompilerMessages()
{
    return "No messages yet";
}

} //namespace rr

