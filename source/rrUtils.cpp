#pragma hdrstop

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

namespace rr
{
using namespace std;

bool cleanFolder(const string& folder, const string& baseName, const std::vector<std::string>& extensions)
{
    for(int i = 0; i < extensions.size(); i++)
       {
        string aFName = joinPath(folder, baseName) + "." + extensions[i];
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

string getMD5(const string& text)
{
    Poco::MD5Engine md5;
    md5.update(text);
    string digestString(Poco::DigestEngine::digestToHex(md5.digest()));
    return digestString;
}

string getTime()
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

string getDateTime()
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

string getTempDir()
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
            Log(Logger::LOG_DEBUG) << "getTempDir(): " << tmpPath;
            return tmpPath;
        } else {
            Log(Logger::LOG_WARNING) << "Temp dir path specified in config, \""
                    << tmpPath << "\" is not a valid path, returning sytem tmp path: "
                    << Poco::Path::temp();
        }
    }

    Poco::Path temp(Poco::Path::temp());
    tmpPath = temp.makeAbsolute().toString();
    Log(Logger::LOG_DEBUG) << "getTempDir(): " << tmpPath;
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
        throw rr::Exception(string("error in GetModuleHandleExA: " + rr::toString(ret)));
    }

    GetModuleFileNameA(hm, buffer, sizeof(buffer));

    string path(buffer);
    return path.substr( 0, path.find_last_of( '\\' ) + 1 );
#endif
}


string getCurrentExeFolder()
{
#if defined(_WIN32) || defined(__WIN32__)
    char path[MAX_PATH];
    if(GetModuleFileNameA(NULL, path, ARRAYSIZE(path)) != 0)
    {
        string aPath(getFilePath(path));
        return aPath;
    }
    return "";
#elif defined(__APPLE__)
    char exepath[PATH_MAX+1] = {0};
    unsigned  bufsize = sizeof(exepath);
    if (_NSGetExecutablePath(exepath, &bufsize) == 0)
    {
        string thePath = getFilePath(exepath);
        Log(lDebug1) << "Current exe folder says:" << thePath;
        return thePath;
    }
    else
    {
        Log(Logger::LOG_ERROR) << "_NSGetExecutablePath failed";
        return "";
    }
#elif defined (__linux)
    char arg1[20];
    char exepath[PATH_MAX+1] = {0};

    sprintf( arg1, "/proc/%d/exe", getpid() );
    ssize_t r = readlink( arg1, exepath, 1024 );

    if (r < 0)
    {
        throw rr::Exception(string("error readlink(") + string((char*)arg1) + string(") failed"));
    }

    string thePath = getFilePath(exepath);
    Log(lDebug1) << "Current exe folder says:" << thePath;
    return thePath;
#endif

}

string getRoadrunnerExe()
{
#if defined(_WIN32) || defined(__WIN32__)
    char path[MAX_PATH];
    if(GetModuleFileNameA(NULL, path, ARRAYSIZE(path)) != 0)
    {
        return path;
    }
    return "";
#elif defined(__APPLE__)
    char exepath[PATH_MAX+1] = {0};
    unsigned  bufsize = sizeof(exepath);
    if (_NSGetExecutablePath(exepath, &bufsize) == 0)
    {
        return exepath;
    }
    else
    {
        Log(Logger::LOG_ERROR) << "_NSGetExecutablePath failed";
        return "";
    }
#elif defined (__linux)
    char arg1[20];
    char exepath[PATH_MAX+1] = {0};

    sprintf( arg1, "/proc/%d/exe", getpid() );
    ssize_t r = readlink( arg1, exepath, 1024 );

    if (r < 0)
    {
        throw rr::Exception(string("error readlink(") + string((char*)arg1) + string(") failed"));
    }

    return exepath;
#endif

}

string getParentFolder(const string& path)
{
    if (path.empty()) {
        return "";
    }

    Poco::Path p(path);
    p = p.makeParent();
    return p.toString();
}

string getCWD()
{
    //Get the working directory
    char *buffer;

    string cwd;
    // Get the current working directory:
    if( (buffer = getcwd( NULL, 512 )) == NULL )
    {
        Log(Logger::LOG_ERROR)<<"getCWD failed";
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

string getFileContent(const string& fName)
{
    string content;

    vector<string> lines = getLinesInFile(fName);
    for(int i = 0; i < lines.size(); i++)
    {
        content += lines[i];
        content += "\n";
    }

    return content;
}

vector<string> getLinesInFile(const string& fName)
{
    vector<string> lines;

    ifstream ifs(fName.c_str());
    if(!ifs)
    {
        Log(Logger::LOG_ERROR)<<"Failed opening file: "<<fName;
        return lines;
    }

    std::string oneLine((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    lines = splitString(oneLine, "\r\n");
    return lines;
}

int indexOf(const std::vector<std::string>& vec, const std::string& elem )
{
    if(!vec.size())
    {
        return -1;
    }
    else
    {
        int index = distance(vec.begin(), find(vec.begin(), vec.end(), elem));
        return index;
    }
}

// String utils
//string RemoveTrailingSeparator(const string& fldr, const char sep = gPathSeparator);//"\\");
string removeTrailingSeparator(const string& _folder, const char sep)
{
    if((_folder.size() > 0) && (_folder[_folder.size() -1] == sep))
    {
        const int endOfPathIndex = _folder.rfind(sep, _folder.size());
        string folder = _folder.substr(0, endOfPathIndex);
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

bool isNullOrEmpty(const string& str)
{
    return !str.size();
}

void pause(bool doIt, const string& msg)
{
    if(!doIt)
    {
        return;
    }

    if(msg.size() == 0)
    {
        cout<<"Hit any key to exit...";
    }
    else
    {
        cout<<msg;
    }
    cin.ignore(0,'\n');

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
bool fileExists(const string& fname, int fileMode)
{
#if !defined (WIN32)
    bool res = (access(fname.c_str(), fileMode) == 0);
#else
    bool res = (_access(fname.c_str(), fileMode) == 0);
#endif

    return res;
}

bool folderExists(const string& folderName)
{
#if defined(WIN32)
    LPCTSTR szPath = folderName.c_str();
    DWORD dwAttrib = GetFileAttributes(szPath);
    return (dwAttrib != INVALID_FILE_ATTRIBUTES && (dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
#else
    struct stat status;
    return stat(folderName.c_str(), &status) == 0 ? true : false;
#endif
}

void createTestSuiteFileNameParts(int caseNr, const string& postFixPart, string& modelFilePath, string& modelName, string& settingsFName)
{
    stringstream modelSubPath;
    stringstream modelFileName;
    stringstream settingsFileName;

    modelSubPath<<setfill('0')<<setw(5)<<caseNr;        //create the "00023" subfolder format
    modelFileName<<setfill('0')<<setw(5)<<caseNr<<postFixPart;
    modelFilePath = joinPath(modelFilePath, modelSubPath.str());
    modelName =  modelFileName.str();
    settingsFileName <<setfill('0')<<setw(5)<<caseNr<<"-settings.txt";
    settingsFName = settingsFileName.str();
}

string getTestSuiteSubFolderName(int caseNr)
{
    stringstream modelSubPath;
    modelSubPath<<setfill('0')<<setw(5)<<caseNr;        //create the "00023" subfolder format
    return modelSubPath.str();
}

bool createFolder(const string& folder)
{
    if(fileExists(folder))
    {
        return true;
    }

#if defined(WIN32)
    int res = mkdir(folder.c_str());
#else
    int temp;
#define MY_MASK 0777
//     printf("Default mask: %o\n", MY_MASK & ~022 & MY_MASK);
      temp = umask(0);
//      printf("Previous umask = %o\n", temp);
    int res = mkdir(folder.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
#endif

    return (res==0) ? true : false;
}

bool createFile(const string& fName, std::ios_base::openmode mode)
{
    ofstream test;
    test.open(fName.c_str(), mode);
    test.close();
    return fileExists(fName);
}

bool copyValues(vector<double>& dest, double* source, const int& nrVals, const int& startIndex)
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

bool copyStdVectorToCArray(const vector<double>& src, double* dest,  int size)
{
    if((size && !dest) || size > src.size())
    {
        Log(Logger::LOG_ERROR)<<"Tried to copy to NULL vector, or incompatible size of vectors";
        return false;
    }

    for(int i = 0; i < size; i++)
    {
        dest[i] = src[i];
    }
    return true;
}

bool copyStdVectorToCArray(const vector<bool>&   src,  bool*  dest,  int size)
{
    if((size && !dest) || size > src.size())
    {
        Log(Logger::LOG_ERROR)<<"Tried to copy to NULL vector, or incompatible size of vectors";
        return false;
    }

    for(int i = 0; i < size; i++)
    {
        dest[i] = src[i];
    }
    return true;
}

vector<double> createVector(const double* src, const int& size)
{
    vector<double> dest;
    if(size && !src)
    {
        Log(Logger::LOG_ERROR)<<"Tried to copy from NULL vector";
        return dest;
    }

    dest.resize(size);
    for(int i = 0; i < size; i++)
    {
        dest[i] = src[i];
    }
    return dest;
}

bool copyCArrayToStdVector(const int* src, vector<int>& dest, int size)
{
    if(size && !src)
    {
        Log(Logger::LOG_ERROR)<<"Tried to copy from NULL vector";
        return false;
    }

    dest.resize(size);
    for(int i = 0; i < size; i++)
    {
        dest[i] = src[i];
    }
    return true;
}

bool copyCArrayToStdVector(const double* src, vector<double>& dest, int size)
{
    if(size && !src)
    {
        Log(Logger::LOG_ERROR)<<"Tried to copy from NULL vector";
        return false;
    }

    dest.resize(size);
    for(int i = 0; i < size; i++)
    {
        dest[i] = src[i];
    }
    return true;
}

bool copyCArrayToStdVector(const bool* src, vector<bool>& dest, int size)
{
    if(size && !src)
    {
        Log(Logger::LOG_ERROR)<<"Tried to copy from NULL vector";
        return false;
    }

    dest.resize(size);
    for(int i = 0; i < size; i++)
    {
        dest[i] = src[i];
    }
    return true;
}

double* createVector(const vector<double>& vec)
{
    double* avec = new double[vec.size()];
    if(!avec)
    {
        Log(Logger::LOG_ERROR)<<"Failed to allocate c vector";
        return NULL;
    }

    for(int i = 0; i < vec.size(); i++)
    {
          avec[i] = vec[i];
    }
    return avec;
}

string joinPath(const string& base, const string& file, const char pathSeparator)
{
    if (base.empty()) {
        return file;
    }

    if (file.empty()) {
        return base;
    }

    Poco::Path basePath(base);
    basePath.append(Poco::Path(file));
    return basePath.toString();
}

string joinPath(const string& p1, const string& p2, const string& p3, const char pathSeparator)
{
    string tmp(joinPath(p1, p2, gPathSeparator));
    return joinPath(tmp, p3, gPathSeparator);
}

string joinPath(const string& p1, const string& p2, const string& p3, const string& p4, const char pathSeparator)
{
    string tmp(joinPath(p1, p2, p3, gPathSeparator));
    return joinPath(tmp, p4, gPathSeparator);
}

string joinPath(const string& p1, const string& p2, const string& p3, const string& p4, const string& p5, const char pathSeparator)
{
    string tmp(joinPath(p1, p2, p3, p4, gPathSeparator));
    return joinPath(tmp, p5, gPathSeparator);
}


#if defined(_WIN32) || defined(__WIN32__)

string getWINAPIError(DWORD errorCode, LPTSTR lpszFunction)
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
        (LPTSTR) &lpMsgBuf,
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

    string errorMsg = string((LPCTSTR)lpDisplayBuf);
    LocalFree(lpMsgBuf);
    LocalFree(lpDisplayBuf);
    return errorMsg;
}

int populateFileSet(const string& folder, set<string>& files)
{
     //Get models file names in models folder
    string globPath =  rr::joinPath(folder, "*.xml");
    Poco::Glob::glob(globPath, files);
    return files.size();
}

#endif

int64_t getMicroSeconds() {
    Poco::Timestamp ts;
    return ts.epochMicroseconds();
}

}//end of namespace
