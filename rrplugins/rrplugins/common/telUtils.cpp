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
#include "telStringUtils.h"
#include "telUtils.h"
#include "../../../source/rrException.h"
//#include "rr/rrException.h"
#include "telException.h"
#include "telLogger.h"

// Most Unix systems have a getch in libcurses, but this introduces
// an un-needed depencency, as we can write our own getch easily.
// We do need the standard Posix termio headers though.
#if defined (__unix__) || defined(__APPLE__)
#include <stdlib.h>
#include <termios.h>
#endif

#if defined(__APPLE__)
#include <limits.h>  //PATH_MAX
#include <mach-o/dyld.h>
#endif

//---------------------------------------------------------------------------
namespace tlp
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

string getUsersTempDataFolder()
{
    //Default for temporary data output is the users AppData/Local/Temp Folder
    //  Gets the temp path env string (no guarantee it's a valid path).
#if defined(WIN32)
    TCHAR lpTempPathBuffer[MAX_PATH];
    DWORD dwRetVal = GetTempPathA(MAX_PATH, lpTempPathBuffer); // buffer for path
    if (dwRetVal > MAX_PATH || (dwRetVal == 0))
    {
        RRPLOG(lError)<<"GetTempPath failed";
    }
    else
    {
        RRPLOG(lDebug3)<<"Users temporary files folder is: "<<string(lpTempPathBuffer);
    }

    return string(lpTempPathBuffer);
#else
return ".";
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
        RRPLOG(lDebug1) << "Current exe folder says:" << thePath;
        return thePath;
    }
    else
    {
        RRPLOG(Logger::LOG_ERROR) << "_NSGetExecutablePath failed";
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
    RRPLOG(lDebug1) << "Current exe folder says:" << thePath;
    return thePath;
#endif

}

string getParentFolder(const string& path)
{
    if(path.size() < 1)
    {
        return "";
    }
    vector<string> fldrs = splitString(path, gPathSeparator);
    string parent("");
    if(fldrs.size() > 1)
    {
        for(int i = 0; i < fldrs.size() -1; i++)
        {
            parent = joinPath(parent, fldrs[i]);
        }

        string pathSep;
        pathSep.push_back(gPathSeparator);
        if(path.compare(0,1, pathSep) == 0)
        {
            parent = gPathSeparator + parent;
        }
        return parent;
    }
    else
    {
        return path;
    }
}

string getCWD()
{
    //Get the working directory
    char *buffer;

    string cwd;
    // Get the current working directory:
    if( (buffer = getcwd( NULL, 512 )) == NULL )
    {
        RRPLOG(Logger::LOG_ERROR)<<"getCWD failed";
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
        RRPLOG(Logger::LOG_ERROR)<<"Failed opening file: "<<fName;
        return lines;
    }

    std::string oneLine((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    lines = splitString(oneLine, "\r\n");
    return lines;
}

ptrdiff_t indexOf(const std::vector<std::string>& vec, const std::string& elem )
{
    if(!vec.size())
    {
        return -1;  
    }
    else
    {
        ptrdiff_t index = distance(vec.begin(), find(vec.begin(), vec.end(), elem));

        if(index > vec.size() - 1)
        {
            return -1;
        }
        return index;
    }
}

// String utils
//string RemoveTrailingSeparator(const string& fldr, const char sep = gPathSeparator);//"\\");
string removeTrailingSeparator(const string& _folder, const char sep)
{
    if((_folder.size() > 0) && (_folder[_folder.size() -1] == sep))
    {
        const size_t endOfPathIndex = _folder.rfind(sep, _folder.size());
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

bool fileExists(const string& fName)
{
    if (!fName.size())
    {
        return false;
    }

    bool res = (access(fName.c_str(), 0) == 0);
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
        RRPLOG(Logger::LOG_ERROR)<<"Tried to copy to NULL vector, or incompatible size of vectors";
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
        RRPLOG(Logger::LOG_ERROR)<<"Tried to copy to NULL vector, or incompatible size of vectors";
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
        RRPLOG(Logger::LOG_ERROR)<<"Tried to copy from NULL vector";
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
        RRPLOG(Logger::LOG_ERROR)<<"Tried to copy from NULL vector";
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
        RRPLOG(Logger::LOG_ERROR)<<"Tried to copy from NULL vector";
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
        RRPLOG(Logger::LOG_ERROR)<<"Tried to copy from NULL vector";
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
        RRPLOG(Logger::LOG_ERROR)<<"Failed to allocate c vector";
        return NULL;
    }

    for(int i = 0; i < vec.size(); i++)
    {
          avec[i] = vec[i];
    }
    return avec;
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

size_t populateFileSet(const string& folder, set<string>& files)
{
     //Get models file names in models folder
    string globPath =  joinPath(folder, "*.xml");
    Poco::Glob::glob(globPath, files);
    return files.size();
}

#endif

}//end of namespace
