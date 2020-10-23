/**
 * @file telUtils.h
 * @brief Utility functions for the plugin framework
 * @author Totte Karlsson & Herbert M Sauro
 *
 * <--------------------------------------------------------------
 * This file is part of cRoadRunner.
 * See http://code.google.com/p/roadrunnerlib for more details.
 *
 * Copyright (C) 2012-2013
 *   University of Washington, Seattle, WA, USA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * In plain english this means:
 *
 * You CAN freely download and use this software, in whole or in part, for personal,
 * company internal, or commercial purposes;
 *
 * You CAN use the software in packages or distributions that you create.
 *
 * You SHOULD include a copy of the license in any redistribution you may make;
 *
 * You are NOT required include the source of software, or of any modifications you may
 * have made to it, in any redistribution you may assemble that includes it.
 *
 * YOU CANNOT:
 *
 * redistribute any piece of this software without proper attribution;
 * <hr>
*/
#ifndef telUtilsH
#define telUtilsH
#if defined(_WIN32)
    #include <windows.h>
#endif

#include <vector>
#include <string>
#include <sstream>
#include <set>
#include "telCommonExporter.h"
#include "telStringList.h"
#include "telUtils.h"
#include "telStringList.h"
#include "telStringUtils.h"
#include <float.h>    //ms compatible IEEE functions, e.g. _isnan
//---------------------------------------------------------------------------

namespace tlp
{

using std::vector;
using std::set;
using std::string;


//enum LogLevel
//{
//    lAny        = rr::Logger::LOG_ERROR,
//    lError      = rr::Logger::LOG_ERROR,
//    lWarning    = rr::Logger::LOG_WARNING,
//    lInfo       = rr::Logger::LOG_INFORMATION,
//    lDebug      = rr::Logger::LOG_DEBUG,
//    lDebug1     = rr::Logger::LOG_TRACE,
//    lDebug2     = rr::Logger::LOG_TRACE,
//    lDebug3     = rr::Logger::LOG_TRACE,
//    lDebug4     = rr::Logger::LOG_TRACE,
//    lDebug5     = rr::Logger::LOG_TRACE,
//    lUser       = rr::Logger::LOG_TRACE
//};
COMMON_DECLSPEC bool           cleanFolder(const string& folder, const string& baseName,  const std::vector<std::string>& extensions);
COMMON_DECLSPEC string         getTime();
COMMON_DECLSPEC string         getDateTime();
COMMON_DECLSPEC string         getMD5(const string& text);
COMMON_DECLSPEC void           sleep(int ms);

//Misc.
/*!
\brief indexOf returns the index of an element in the vector. If not found, or if vector size is zero, it returns -1.
*/
COMMON_DECLSPEC ptrdiff_t       indexOf(const std::vector<std::string>& vec, const std::string& elem );
COMMON_DECLSPEC bool            isNaN(const double& aNum);
COMMON_DECLSPEC bool            isNullOrEmpty(const string& str);    //Can't be null, but empty
COMMON_DECLSPEC void            pause(bool doIt = true, const string& msg = "");

//String utilities
COMMON_DECLSPEC string          removeTrailingSeparator(const string& fldr, const char sep = gPathSeparator);//"\\");

//File  Utilities
COMMON_DECLSPEC size_t          populateFileSet(const string& modelsFolder, set<string>& models);
COMMON_DECLSPEC bool            fileExists(const string& fileN);
COMMON_DECLSPEC bool            folderExists(const string& folderN);
COMMON_DECLSPEC bool            createFolder(const string& path);

COMMON_DECLSPEC string          getParentFolder(const string& path);
COMMON_DECLSPEC string          getCurrentExeFolder();
COMMON_DECLSPEC string          getUsersTempDataFolder();
COMMON_DECLSPEC string          getCWD();
COMMON_DECLSPEC const char      getPathSeparator();

COMMON_DECLSPEC vector<string>  getLinesInFile(const string& fName);
COMMON_DECLSPEC string          getFileContent(const string& fName);
COMMON_DECLSPEC void            createTestSuiteFileNameParts(int caseNr, const string& postFixPart, string& FilePath, string& modelFileName, string& settingsFileName);
COMMON_DECLSPEC string          getTestSuiteSubFolderName(int caseNr);

//CArray utilities
COMMON_DECLSPEC bool            copyCArrayToStdVector(const int* src,     vector<int>& dest, int size);
COMMON_DECLSPEC bool            copyCArrayToStdVector(const double* src,  vector<double>& dest, int size);
COMMON_DECLSPEC bool            copyValues(vector<double>& dest, double* source, const int& nrVals, const int& startIndex);
COMMON_DECLSPEC bool            copyCArrayToStdVector(const bool* src,    vector<bool>& dest, int size);
COMMON_DECLSPEC bool            copyStdVectorToCArray(const vector<double>& src, double* dest,  int size);
COMMON_DECLSPEC bool            copyStdVectorToCArray(const vector<bool>&   src,  bool*  dest,  int size);
COMMON_DECLSPEC double*         createVector(const vector<double>& vec);
COMMON_DECLSPEC vector<double>  createVector(const double* src, const int& size);

#if defined(_WIN32) || defined(__WIN32__)
COMMON_DECLSPEC HINSTANCE       loadDLL(const string& dll);
COMMON_DECLSPEC bool            unLoadDLL(HINSTANCE dllHandle);
COMMON_DECLSPEC FARPROC         getFunctionPtr(const string& funcName, HINSTANCE DLLHandle);
COMMON_DECLSPEC string          getWINAPIError(DWORD errorCode, LPTSTR lpszFunction);
#endif

#undef CreateFile
COMMON_DECLSPEC bool            createFile(const string& fName, std::ios_base::openmode mode = std::ios::trunc );

} // rr Namespace
#endif
