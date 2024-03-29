#pragma hdrstop

#include <filesystem>
#if defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#include <io.h>
#include <conio.h>
#endif

#if defined(__BORLANDC__)
#include <dir.h>
#elif defined(_MSC_VER)
#include <direct.h>
#else
#include <unistd.h>
#include <sys/stat.h>
#include <limits.h>  //PATH_MAX
#endif

#define _USE_MATH_DEFINES

#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <math.h>
#include <float.h>
#include <fstream>
#if defined(_WIN32)
#include <strsafe.h> //StringCchPrintf need to be included AFTER all other system headers :(
#endif
#include "Poco/MD5Engine.h"
#include "Poco/Thread.h"
#include "Poco/Glob.h"
#include "Poco/File.h"
#include <Poco/Path.h>
#include "rrStringUtils.h"
#include "rrUtils.h"
#include "rrLogger.h"
#include "rrSelectionRecord.h"
#include "rrException.h"
#include "rrConfig.h"
#include "rrRoadRunnerOptions.h"

// Most Unix systems have a getch in libcurses, but this introduces
// an un-needed depencency, as we can write our own getch easily.
// We do need the standard Posix termio headers though.
#if defined (__unix__) || defined(__APPLE__)
#include <stdlib.h>
#include <termios.h>
#include <dlfcn.h>
#endif

#if defined(__APPLE__)
#include <limits.h>  //PATH_MAX
#include <mach-o/dyld.h>
#endif

// Poco timestamp
#include <Poco/Timestamp.h>

using std::filesystem::path;
namespace rr
{


bool cleanFolder(const std::string& folder, const std::string& baseName, const std::vector<std::string>& extensions)
{
    for(int i = 0; i < extensions.size(); i++)
       {
        std::string aFName = (path(folder) /= baseName).string() + "." + extensions[i];
        Poco::File aFile(aFName);
        if(aFile.exists())
        {
            aFile.remove();
        }
    }
    return true;
}


// A function to get a character from the console without echo.
// equivalent of Windows / Curses getch function. Note, that the
// curses library has the same thing, but not all systems have curses,
// and makes no sense have a dependency on it for one simple function.
#if defined(__unix__) || defined(__APPLE__)
static char rrGetch()
{
    char ch;
    termios _old, _new;

    /* Initialize new terminal i/o settings */
    tcgetattr(0, &_old); /* grab old terminal i/o settings */
    _new = _old; /* make new settings same as old settings */
    _new.c_lflag &= ~ICANON; /* disable buffered i/o */
    _new.c_lflag &= ~ECHO; /* set no echo mode */
    tcsetattr(0, TCSANOW, &_new); /* use these new terminal i/o settings now */

    ch = getchar();

    /* Restore old terminal i/o settings */
    tcsetattr(0, TCSANOW, &_old);
    return ch;
}
#elif defined (_WIN32)
// Windows has get built into conio
#define rrGetch getch
#endif

// todo switch to using llvm::MD5Hash()
//  -- reduce dependency on poco
std::string getMD5(const std::string& text)
{
    Poco::MD5Engine md5;
    md5.update(text);
    std::string digestString(Poco::DigestEngine::digestToHex(md5.digest()));
    return digestString;
}

std::string getSBMLMD5(const std::string& text, int modelGenOpt)
{
    std::string digestString = getMD5(text);
    if (modelGenOpt & rr::LoadSBMLOptions::CONSERVED_MOIETIES) {
        digestString += "_conserved";
    }
    return digestString;
}

std::string getTime()
{
// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://www.cplusplus.com/reference/clibrary/ctime/strftime/
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%X", &tstruct);

    return buf;
}

std::string getDateTime()
{
    // Get current date/time, format is YYYY-MM-DD.HH:mm:ss
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://www.cplusplus.com/reference/clibrary/ctime/strftime/
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d %X", &tstruct);

    return buf;
}

std::string getTempDir()
{
    std::string tmpPath = Config::getString(Config::TEMP_DIR_PATH);

    if(tmpPath.length() > 0) {
        tmpPath = Poco::Path::expand(tmpPath);
        Poco::Path path(tmpPath);
        path = path.makeAbsolute();
        path = path.makeDirectory();
        Poco::File file(path);

        if (file.exists() && file.isDirectory()) {
            tmpPath = path.toString();
            rrLog(Logger::LOG_DEBUG) << "getTempDir(): " << tmpPath;
            return tmpPath;
        } else {
            rrLog(Logger::LOG_WARNING) << "Temp dir path specified in config, \""
                    << tmpPath << "\" is not a valid path, returning sytem tmp path: "
                    << Poco::Path::temp();
        }
    }

    Poco::Path temp(Poco::Path::temp());
    tmpPath = temp.makeAbsolute().toString();
    rrLog(Logger::LOG_DEBUG) << "getTempDir(): " << tmpPath;
    return tmpPath;
}

std::string getCurrentSharedLibDir()
{
#if !defined(_WIN32)
    Dl_info dl_info;

    if (dladdr((void *)getCurrentSharedLibDir, &dl_info) != 0) {

        // on Linux, either returns
        // 1: absolute path if actually a shared library
        // 2: "./" if run from current directory   - if statically linked
        // 3: "exename" if run from path           - if statically linked
        Poco::Path p(dl_info.dli_fname);

        if (Poco::File(p).exists()) {
            return p.parent().toString();
        }
    }

    return "";


#else
    char buffer[MAX_PATH] = {0};
    HMODULE hm = NULL;

    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            (LPCSTR) &getCurrentSharedLibDir,
            &hm))
    {
        int ret = GetLastError();
        throw rr::Exception(std::string("error in GetModuleHandleExA: " + rr::toString(ret)));
    }

    GetModuleFileNameA(hm, buffer, sizeof(buffer));

    std::string path(buffer);
    return path.substr( 0, path.find_last_of( '\\' ) + 1 );
#endif
}


std::string getCurrentExeFolder()
{
#if defined(_WIN32) || defined(__WIN32__)
    char path[MAX_PATH];
    if(GetModuleFileNameA(NULL, path, ARRAYSIZE(path)) != 0)
    {
        std::string aPath(getFilePath(path));
        return aPath;
    }
    return "";
#elif defined(__APPLE__)
    char exepath[PATH_MAX+1] = {0};
    unsigned  bufsize = sizeof(exepath);
    if (_NSGetExecutablePath(exepath, &bufsize) == 0)
    {
        std::string thePath = getFilePath(exepath);
        rrLog(lDebug1) << "Current exe folder says:" << thePath;
        return thePath;
    }
    else
    {
        rrLog(Logger::LOG_ERROR) << "_NSGetExecutablePath failed";
        return "";
    }
#elif defined (__linux)
    char arg1[20];
    char exepath[PATH_MAX+1] = {0};

    sprintf( arg1, "/proc/%d/exe", getpid() );
    ssize_t r = readlink( arg1, exepath, 1024 );

    if (r < 0)
    {
        throw rr::Exception(std::string("error readlink(") + std::string((char*)arg1) + std::string(") failed"));
    }

    std::string thePath = getFilePath(exepath);
    rrLog(lDebug1) << "Current exe folder says:" << thePath;
    return thePath;
#endif

}

std::string getParentFolder(const std::string& path)
{
    if (path.empty()) {
        return "";
    }

    Poco::Path p(path);
    p = p.makeParent();
    return p.toString();
}

std::string getCWD()
{
    //Get the working directory
    char *buffer;

    std::string cwd;
    // Get the current working directory:
    if( (buffer = getcwd( NULL, 512 )) == NULL )
    {
        rrLog(Logger::LOG_ERROR)<<"getCWD failed";
        return "";
    }
    else
    {
      cwd = buffer;
      free(buffer);
    }

    return cwd;
}

void sleep(int milliSeconds)
{
    Poco::Thread::sleep(milliSeconds);
}

const char getPathSeparator()
{
    return gPathSeparator;
}

std::string getFileContent(std::filesystem::path fName)
{
    std::string content;

    std::vector<std::string> lines = getLinesInFile(fName);
    for(int i = 0; i < lines.size(); i++)
    {
        content += lines[i];
        content += "\n";
    }

    return content;
}

std::vector<std::string> getLinesInFile(std::filesystem::path fName)
{
    std::vector<std::string> lines;

    std::ifstream ifs(fName.c_str());
    if(!ifs)
    {
        rrLog(Logger::LOG_ERROR)<<"Failed opening file: "<<fName;
        return lines;
    }

    std::string oneLine((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    lines = splitString(oneLine, "\r\n");
    return lines;
}

std::ptrdiff_t indexOf(const std::vector<std::string>& vec, const std::string& elem )
{
    if(!vec.size())
    {
        return -1;
    }
    else
    {
        std::ptrdiff_t index = distance(vec.begin(), find(vec.begin(), vec.end(), elem));
        return index;
    }
}

// String utils
//std::string RemoveTrailingSeparator(const std::string& fldr, const char sep = gPathSeparator);//"\\");
std::string removeTrailingSeparator(const std::string& _folder, const char sep)
{
    if((_folder.size() > 0) && (_folder[_folder.size() -1] == sep))
    {
        size_t endOfPathIndex = _folder.rfind(sep, _folder.size());
        std::string folder = _folder.substr(0, endOfPathIndex);
        return folder;
    }
    else
    {
        return _folder;
    }
}

bool isNaN(const double& aNum)
{
#if defined(WIN32)
    return _isnan(aNum) > 0 ? true : false;
#else
    return std::isnan(aNum) > 0 ? true : false;
#endif
}

bool isNullOrEmpty(const std::string& str)
{
    return !str.size();
}

void pause(bool doIt, const std::string& msg)
{
    if(!doIt)
    {
        return;
    }

    if(msg.size() == 0)
    {
        std::cout<<"Hit any key to exit...";
    }
    else
    {
        std::cout<<msg;
    }
    std::cin.ignore(0,'\n');

    rrGetch();


}

/**
 * check file access, just calls access, but on windows, access is _access,
 *
 * file modes:
 * 00 Existence only
 * 02 Write permission
 * 04 Read permission
 * 06 Read and write permission
*/
//bool fileExists(const std::string& fname, int fileMode)
//{
//#if !defined (WIN32)
//    bool res = (access(fname.c_str(), fileMode) == 0);
//#else
//    bool res = (_access(fname.c_str(), fileMode) == 0);
//#endif
//
//    return res;
//}

//bool folderExists(const std::string& folderName)
//{
//    return std::filesystem::exists(folderName);
//#if defined(WIN32)
//    LPCTSTR szPath = folderName.c_str();
//    DWORD dwAttrib = GetFileAttributes(szPath);
//    return (dwAttrib != INVALID_FILE_ATTRIBUTES && (dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
//#else
//    struct stat status;
//    return stat(folderName.c_str(), &status) == 0 ? true : false;
//#endif
//}

void createTestSuiteFileNameParts(int caseNr, const std::string& postFixPart, std::string & modelFilePath, std::string& modelName, std::string& settingsFName, std::string& descriptionFName)
{
    std::stringstream modelSubPath, modelFileName, settingsFileName, descriptionFileName;

    modelSubPath<<std::setfill('0')<<std::setw(5)<<caseNr;        //create the "00023" subfolder format
    modelFileName<<std::setfill('0')<<std::setw(5)<<caseNr<<postFixPart;
    modelFilePath = (path(modelFilePath) /= modelSubPath.str()).string();
    modelName =  modelFileName.str();
    settingsFileName << std::setfill('0')<<std::setw(5)<<caseNr<<"-settings.txt";
    settingsFName = settingsFileName.str();
    descriptionFileName << std::setfill('0') << std::setw(5) << caseNr << "-model.m";
    descriptionFName = descriptionFileName.str();
}

std::string getTestSuiteSubFolderName(int caseNr)
{
    std::stringstream modelSubPath;
    modelSubPath<<std::setfill('0')<<std::setw(5)<<caseNr;        //create the "00023" subfolder format
    return modelSubPath.str();
}

bool hasUnimplementedTags(const std::string& descriptionFileName, const string& integrator)
{
    std::vector<std::string> badtags;
    badtags.push_back("AlgebraicRule");
    badtags.push_back("CSymbolDelay");
    badtags.push_back("fbc");
    badtags.push_back("FastReaction");
    badtags.push_back("VolumeConcentrationRate");
    //badtags.push_back("RateOf");
    badtags.push_back("AssignedVariableStoichiometry");

    if (integrator == "rk4" || integrator == "rk45")
    {
        badtags.push_back("EventWithDelay");
        badtags.push_back("EventNoDelay");
    }
    std::ifstream descfile(descriptionFileName);
    if (descfile.good()) {
        std::string line;
        while (getline(descfile, line)) {
            if (line.find("Tags") != std::string::npos) {
                for (size_t i = 0; i < badtags.size(); i++) {
                    std::string tag = badtags[i];
                    if (line.find(tag) != std::string::npos) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

bool isSemiStochasticTest(const std::string& descriptionFileName)
{
    std::ifstream descfile(descriptionFileName);
    if (descfile.good()) {
        std::string line;
        while (getline(descfile, line)) {
            if (line.find("synopsis") != std::string::npos && line.find("STOCHASTIC") != std::string::npos) {
                return true;
            }
        }
    }
    return false;
}

bool isFBCTest(const std::string& descriptionFileName)
{
    std::ifstream descfile(descriptionFileName);
    if (descfile.good()) {
        std::string line;
        while (getline(descfile, line)) {
            if (line.find("testType") != std::string::npos) {
                if (line.find("FluxBalanceSteadyState") != std::string::npos) {
                    return true;
                }
                else {
                    return false;
                }
            }
        }
    }
    return false;
}

//bool createFolder(const std::string& folder)
//{
//    if(std::filesystem::exists(folder))
//    {
//        return true;
//    }
//
//#if defined(WIN32)
//    int res = mkdir(folder.c_str());
//#else
//    int temp;
//#define MY_MASK 0777
////     printf("Default mask: %o\n", MY_MASK & ~022 & MY_MASK);
//      temp = umask(0);
////      printf("Previous umask = %o\n", temp);
//    int res = mkdir(folder.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
//#endif
//
//    return (res==0) ? true : false;
//}

bool createFolder(const std::string& folder)
{
    if(std::filesystem::exists(folder))
    {
        return true;
    }
    try {
        std::filesystem::create_directories(folder);
        return true;
    } catch (std::exception &e) {
        return false;
    }
}

bool createFile(const std::string& fName, std::ios_base::openmode mode)
{
    std::ofstream test;
    test.open(fName.c_str(), mode);
    test.close();
    return std::filesystem::exists(fName);
}

bool copyValues(std::vector<double>& dest, double* source, const int& nrVals, const int& startIndex)
{
    if(!dest.size() || !source || startIndex > dest.size())
    {
        return false;
    }

    for(int i = startIndex; i < nrVals; i++)
    {
        dest[i] = source[i];
    }

    return true;

}

bool copyStdVectorToCArray(const std::vector<double>& src, double* dest,  int size)
{
    if((size && !dest) || size > src.size())
    {
        rrLog(Logger::LOG_ERROR)<<"Tried to copy to NULL std::vector, or incompatible size of vectors";
        return false;
    }

    for(int i = 0; i < size; i++)
    {
        dest[i] = src[i];
    }
    return true;
}

bool copyStdVectorToCArray(const std::vector<bool>&   src,  bool*  dest,  int size)
{
    if((size && !dest) || size > src.size())
    {
        rrLog(Logger::LOG_ERROR)<<"Tried to copy to NULL std::vector, or incompatible size of vectors";
        return false;
    }

    for(int i = 0; i < size; i++)
    {
        dest[i] = src[i];
    }
    return true;
}

std::vector<double> createVector(const double* src, const int& size)
{
    std::vector<double> dest;
    if(size && !src)
    {
        rrLog(Logger::LOG_ERROR)<<"Tried to copy from NULL std::vector";
        return dest;
    }

    dest.resize(size);
    for(int i = 0; i < size; i++)
    {
        dest[i] = src[i];
    }
    return dest;
}

bool copyCArrayToStdVector(const int* src, std::vector<int>& dest, int size)
{
    if(size && !src)
    {
        rrLog(Logger::LOG_ERROR)<<"Tried to copy from NULL std::vector";
        return false;
    }

    dest.resize(size);
    for(int i = 0; i < size; i++)
    {
        dest[i] = src[i];
    }
    return true;
}

bool copyCArrayToStdVector(const double* src, std::vector<double>& dest, int size)
{
    if(size && !src)
    {
        rrLog(Logger::LOG_ERROR)<<"Tried to copy from NULL std::vector";
        return false;
    }

    dest.resize(size);
    for(int i = 0; i < size; i++)
    {
        dest[i] = src[i];
    }
    return true;
}

bool copyCArrayToStdVector(const bool* src, std::vector<bool>& dest, int size)
{
    if(size && !src)
    {
        rrLog(Logger::LOG_ERROR)<<"Tried to copy from NULL std::vector";
        return false;
    }

    dest.resize(size);
    for(int i = 0; i < size; i++)
    {
        dest[i] = src[i];
    }
    return true;
}

double* createVector(const std::vector<double>& vec)
{
    double* avec = new double[vec.size()];
    if(!avec)
    {
        rrLog(Logger::LOG_ERROR)<<"Failed to allocate c std::vector";
        return NULL;
    }

    for(int i = 0; i < vec.size(); i++)
    {
          avec[i] = vec[i];
    }
    return avec;
}
//
//std::string joinPath(const std::string& base, const std::string& file, const char pathSeparator)
//{
//    if (base.empty()) {
//        return file;
//    }
//
//    if (file.empty()) {
//        return base;
//    }
//
//    Poco::Path basePath(base);
//    basePath.append(Poco::Path(file));
//    return basePath.toString();
//}
//
//std::string joinPath(const std::string& p1, const std::string& p2, const std::string& p3, const char pathSeparator)
//{
//    std::string tmp(joinPath(p1, p2, gPathSeparator));
//    return joinPath(tmp, p3, gPathSeparator);
//}
//
//std::string joinPath(const std::string& p1, const std::string& p2, const std::string& p3, const std::string& p4, const char pathSeparator)
//{
//    std::string tmp(joinPath(p1, p2, p3, gPathSeparator));
//    return joinPath(tmp, p4, gPathSeparator);
//}
//
//std::string joinPath(const std::string& p1, const std::string& p2, const std::string& p3, const std::string& p4, const std::string& p5, const char pathSeparator)
//{
//    std::string tmp(joinPath(p1, p2, p3, p4, gPathSeparator));
//    return joinPath(tmp, p5, gPathSeparator);
//}


#if defined(_WIN32) || defined(__WIN32__)

std::string getWINAPIError(DWORD errorCode, LPTSTR lpszFunction)
{
    LPVOID lpMsgBuf;
    LPVOID lpDisplayBuf;
    DWORD dw = GetLastError();

    FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER |
        FORMAT_MESSAGE_FROM_SYSTEM |
        FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        dw,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPSTR) &lpMsgBuf,
        0, NULL
    );

    // Display the error message
    lpDisplayBuf = (LPVOID)LocalAlloc(LMEM_ZEROINIT, (lstrlen((LPCTSTR)lpMsgBuf) + lstrlen((LPCTSTR)lpszFunction) + 40) * sizeof(TCHAR));

    StringCchPrintf((LPTSTR)lpDisplayBuf,
                        LocalSize(lpDisplayBuf) / sizeof(TCHAR),
                        TEXT("%s failed with error %d: %s"),
                        lpszFunction,
                        dw,
                        lpMsgBuf);

	// This used to be cast to LPCTSTR, but it didn't compile
    std::string errorMsg = std::string((const char*)lpDisplayBuf);
    LocalFree(lpMsgBuf);
    LocalFree(lpDisplayBuf);
    return errorMsg;
}

size_t populateFileSet(const std::string& folder, std::set<std::string>& files)
{
     //Get models file names in models folder
    std::string globPath =  (path(folder) /= "*.xml").string();
    Poco::Glob::glob(globPath, files);
    return files.size();
}

#endif

int64_t getMicroSeconds() {
    Poco::Timestamp ts;
    return ts.epochMicroseconds();
}

}//end of namespace
