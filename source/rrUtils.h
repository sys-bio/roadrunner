#ifndef rrUtilsH
#define rrUtilsH

#if defined(_WIN32) || defined(__WIN32__)
#define NOMINMAX
#include <windows.h>
#endif

#include <float.h>    //ms compatible IEEE functions, e.g. _isnan
#include <vector>
#include <string>
#include <iostream>
#include <set>
#include "rrExporter.h"
#include "rrConstants.h"
#include "rrStringUtils.h"
#include <stdint.h>
#include <filesystem>


namespace rr
{

using std::vector;
using std::string;
using std::set;
//namespace fs = std::filesystem ;

RR_DECLSPEC bool                 cleanFolder(const std::string& folder, const std::string& baseName,  const std::vector<std::string>& extensions);
RR_DECLSPEC std::string          getTime();
RR_DECLSPEC std::string          getDateTime();
RR_DECLSPEC std::string          getMD5(const std::string& text);
RR_DECLSPEC void                 sleep(int ms);

//Misc.
/*!
\brief indexOf returns the index of an element in the std::vector. If not found, or if std::vector size is zero, it returns -1.
*/
RR_DECLSPEC std::ptrdiff_t       indexOf(const std::vector<std::string>& vec, const std::string& elem );
RR_DECLSPEC bool            isNaN(const double& aNum);
RR_DECLSPEC bool            isNullOrEmpty(const std::string& str);    //Can't be null, but empty
RR_DECLSPEC void            pause(bool doIt = true, const std::string& msg = "");

//String utilities
RR_DECLSPEC std::string          removeTrailingSeparator(const std::string& fldr, const char sep = gPathSeparator);//"\\");

//File  Utilities
RR_DECLSPEC size_t          populateFileSet(const std::string& modelsFolder, std::set<std::string>& models);


/**
 * check file access, just calls access, but on windows, access is _access,
 *
 * file modes:
 * 00 Existence only
 * 02 Write permission
 * 04 Read permission
 * 06 Read and write permission
*/
//RR_DECLSPEC bool fileExists(const std::string& fileN, int fileMode=0);
//
//RR_DECLSPEC bool            folderExists(const std::string& folderN);
RR_DECLSPEC bool            createFolder(const std::string& path);

RR_DECLSPEC std::string          getParentFolder(const std::string& path);
RR_DECLSPEC std::string          getCurrentExeFolder();

/**
 * If roadrunner is build and running as shared library,
 * either using the C++ shared library, or using the
 * _roadrunner.so (or .pyd on Win).
 */
RR_DECLSPEC std::string getCurrentSharedLibDir();

/**
 * Get a temporary directory path.
 *
 * If the config specifies a TEMP_DIR entry, this is used, if not
 * this will return the system specified temporary directory.
 */
RR_DECLSPEC std::string getTempDir();

///**
// * Join two file paths together.
// */
//RR_DECLSPEC std::string joinPath(const std::string& p1, const std::string& p2,
//        const char pathSeparator = gPathSeparator);
//
///**
// * join three file paths together.
// */
//RR_DECLSPEC std::string joinPath(const std::string& p1, const std::string& p2,
//        const std::string& p3, const char pathSeparator = gPathSeparator);
//
//RR_DECLSPEC std::string joinPath(const std::string& p1, const std::string& p2,
//        const std::string& p3, const std::string& p4, const char pathSeparator =
//                gPathSeparator);
//
//RR_DECLSPEC std::string joinPath(const std::string& p1, const std::string& p2,
//        const std::string& p3, const std::string& p4, const std::string& p5,
//        const char pathSeparator = gPathSeparator);


RR_DECLSPEC std::string          getCWD();
RR_DECLSPEC const char      getPathSeparator();

RR_DECLSPEC std::vector<std::string>    getLinesInFile(std::filesystem::path fName);
RR_DECLSPEC std::string                 getFileContent(std::filesystem::path fName);
RR_DECLSPEC void                        createTestSuiteFileNameParts(int caseNr, const std::string& postFixPart, std::string& FilePath, std::string& modelFileName, std::string& settingsFileName, std::string& descriptionFileName);
RR_DECLSPEC std::string                 getTestSuiteSubFolderName(int caseNr);

RR_DECLSPEC bool            hasUnimplementedTags(const std::string& descriptionFileName, const std::string& integrator);

RR_DECLSPEC bool            isSemiStochasticTest(const std::string& descriptionFileName);

RR_DECLSPEC bool            isFBCTest(const std::string& descriptionFileName);

//CArray utilities
RR_DECLSPEC bool            copyCArrayToStdVector(const int* src,     std::vector<int>& dest, int size);
RR_DECLSPEC bool            copyCArrayToStdVector(const double* src,  std::vector<double>& dest, int size);
RR_DECLSPEC bool            copyValues(std::vector<double>& dest, double* source, const int& nrVals, const int& startIndex);
RR_DECLSPEC bool            copyCArrayToStdVector(const bool* src,    std::vector<bool>& dest, int size);
RR_DECLSPEC bool            copyStdVectorToCArray(const std::vector<double>& src, double* dest,  int size);
RR_DECLSPEC bool            copyStdVectorToCArray(const std::vector<bool>&   src,  bool*  dest,  int size);
RR_DECLSPEC double*         createVector(const std::vector<double>& vec);
RR_DECLSPEC std::vector<double>  createVector(const double* src, const int& size);

#if defined(_WIN32) || defined(__WIN32__)
RR_DECLSPEC HINSTANCE       loadDLL(const std::string& dll);
RR_DECLSPEC bool            unLoadDLL(HINSTANCE dllHandle);
RR_DECLSPEC FARPROC         getFunctionPtr(const std::string& funcName, HINSTANCE DLLHandle);
RR_DECLSPEC std::string          getWINAPIError(DWORD errorCode, LPTSTR lpszFunction);
#endif

#undef CreateFile
RR_DECLSPEC bool            createFile(const std::string& fName, std::ios_base::openmode mode = std::ios::trunc );



/**
 * Returns the timestamp expressed in microseconds since the Unix epoch, midnight, January 1, 1970.
 *
 * Note, this is a *PRIVATE* function, only reason why it is exported is so that
 * the test suite can verify that it is increasing.
 */
RR_DECLSPEC int64_t getMicroSeconds();

} // rr Namespace
#endif
