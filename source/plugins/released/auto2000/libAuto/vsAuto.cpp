#include <iostream>
#include <cstring>
#include <fstream>

#if defined(_MSC_VER)
    #include <direct.h>
    #define getcwd _getcwd
    #define chdir  _chdir
#elif defined(__CODEGEARC__)
      #include <dir.h>
#else
    #include <unistd.h>
#endif

#include "vsAuto.h"

FuncCallBack callbackFunc = NULL;
FuncCallBack2 callbackFunc2 = NULL;
StpntCallBack callbackStpnt = NULL;
BcndCallBack callbackBcnd = NULL;
IcndCallBack callbackIcnd = NULL;
FoptCallBack callbackFopt = NULL;
PvlsCallBack callbackPvls = NULL;


namespace autolib
{
using namespace std;

void CloseAllFiles();
int AUTO_main(int argc,char *argv[]);

extern int num_model_pars;
extern int num_total_pars;
extern int sysoff;

//char *_sFort2 = NULL;
//char *_sFort3 = NULL;
//int _nFort2Length;
//int _nFort3Length;

int func (integer ndim, const doublereal *orig_u, const integer *icp,
          const doublereal *orig_par, integer ijac,
          doublereal *orig_f, doublereal *dfdu, doublereal *dfdp)
{

    doublereal* u = (doublereal*)malloc(sizeof(doublereal)*ndim);
    memcpy(u, orig_u, sizeof(doublereal)*ndim);

    doublereal* par = (doublereal*)malloc(sizeof(doublereal)*10);
    memcpy(par, orig_par, sizeof(doublereal)*10);

    doublereal* f = (doublereal*)malloc(sizeof(doublereal)*ndim);
    memset(f, 0, sizeof(doublereal)*ndim);

    if (true)
    {
        if (callbackFunc != NULL)
        {
            callbackFunc(ndim, u, icp, par, ijac, f, dfdu, dfdp);
        }

        if (callbackFunc2 != NULL)
        {
            const FuncCallBack2 oTemp = callbackFunc2;
            oTemp(u, par, f);
        }
    }

    memcpy(orig_f, f, sizeof(doublereal)*ndim);

    free(f);
    free(par);
    free(u);
    return 0;
}

void autoCallConv setCallbackStpnt(StpntCallBack cb)
{
    callbackStpnt = cb;
}

void autoCallConv setCallbackFunc(FuncCallBack cb)
{
    callbackFunc = cb;
}

void autoCallConv setCallbackFunc2(FuncCallBack2 cb)
{
    callbackFunc2 = cb;
}

int stpnt (integer ndim, doublereal t, doublereal *u, doublereal *par)
{
    if (callbackStpnt != NULL)
        return (*callbackStpnt)(ndim, t, u, par);
    return 0;
}

string autoCallConv GetTempPath()
{
#if defined(_WIN32) || defined(__WIN32__)
//    char *tempVar = getenv("TEMP");
//    if (tempVar  == NULL)
//    {
//        tempVar = "c:\\Windows\\Temp";
//    }
    return "r:\\temp";//strdup(tempVar);
#else
    char* path = (char*)malloc(sizeof(char)*5);
    strcpy(path, "/tmp\0");
    return path;
#endif
}

string autoCallConv getFullPath(const string& fileName)
{
    string tempPath =  "r:\\temp";//GetTempPath();
    string result = tempPath;
    result +="\\";
    result += fileName;//(char*) malloc(sizeof(char)*(strlen(tempPath) + strlen(fileName)));
//    memset(result,0,sizeof(char)*(strlen(tempPath) + strlen(fileName)));
//    strcat(result, tempPath);
//    strcat(result, fileName);
    //free(tempPath);
    return result;
}

int bcnd (integer ndim, const doublereal *par, const integer *icp,
          integer nbc, const doublereal *u0, const doublereal *u1, integer ijac,
          doublereal *fb, doublereal *dbc)
{
    if (callbackBcnd != NULL)
        return (*callbackBcnd)(ndim, par, icp, nbc, u0, u1, ijac, fb, dbc);
    return 0;
}

void autoCallConv setCallbackBcnd(BcndCallBack cb)
{
    callbackBcnd = cb;
}


int icnd (integer ndim, const doublereal *par, const integer *icp,
          integer nint, const doublereal *u, const doublereal *uold,
          const doublereal *udot, const doublereal *upold, integer ijac,
          doublereal *fi, doublereal *dint)
{
    if (callbackIcnd != NULL)
        return (*callbackIcnd)(ndim, par, icp, nint, u, uold, udot, upold, ijac, fi, dint);
    return 0;
}

void autoCallConv setCallbackIcnd(IcndCallBack cb)
{
    callbackIcnd = cb;
}

int fopt (integer ndim, const doublereal *u, const integer *icp,
          const doublereal *par, integer ijac,
          doublereal *fs, doublereal *dfdu, doublereal *dfdp)
{
    if (callbackFopt != NULL)
        return (*callbackFopt)(ndim, u, icp, par, ijac, fs, dfdu, dfdp);
    return 0;
}

void autoCallConv setCallbackFopt(FoptCallBack cb)
{
    callbackFopt = cb;
}

int pvls (integer ndim, const void *u,
          doublereal *par)
{
    if (callbackPvls != NULL)
        return (*callbackPvls)(ndim, u, par);

    return 0;
}

void autoCallConv setCallbackPvls(PvlsCallBack cb)
{
    callbackPvls = cb;
}

void clearCallbacks()
{
    callbackBcnd = NULL;
    callbackFopt = NULL;
    callbackFunc = NULL;
    callbackIcnd = NULL;
    callbackPvls = NULL;
    callbackStpnt = NULL;
}

void autoCallConv ResetAutoLib()
{
    clearCallbacks();
}

void autoCallConv SetAutoNumParameters(int n)
{
    num_model_pars = n;
    num_total_pars = 4*n;
    sysoff = n;
}

void autoCallConv createFort2File(const string& content, const string& fName)
{
    ofstream f(fName.c_str());
    f<<content;
    f.close();
}

void autoCallConv createFort3File(const string& content, const string& fName)
{
    ofstream f(fName.c_str());
    f<<content;
    f.close();
}

char *_sFort7 = NULL;
char *_sFort8 = NULL;
char *_sFort9 = NULL;
int _nFort7Length;
int _nFort8Length;

char* autoCallConv getFort7File(int length)
{
    if (_sFort7 != NULL)
        free(_sFort7);

    length = 0;

    string sFileName(GetTempPath());
    sFileName = sFileName + "fort.7";

    FILE* fp = fopen(sFileName.c_str(), "rb");
    if (fp == NULL) return NULL;
    fseek(fp,0,SEEK_END);
    length=ftell(fp);
    fseek(fp,0,SEEK_SET);
    _sFort7=(char *)malloc(length);
    fread(_sFort7,length,1,fp);
    fclose(fp);

    return _sFort7;
}

void autoCallConv setFort7File(char* content, int length)
{
    string fileName = getFullPath("fort.7");

    if (_sFort7 != NULL)
        free (_sFort7);

    _sFort7 = strdup(content);
    _nFort7Length = length;

    ofstream f(fileName.c_str());
    f<<content;
    f.close();
}

char* autoCallConv getFort8File(int length)
{
    if (_sFort8 != NULL)
        free(_sFort8);

    length = 0;

    string sFileName(GetTempPath());
    sFileName = sFileName + "fort.8";

    FILE* fp = fopen(sFileName.c_str(), "rb");
    if (fp == NULL) return NULL;
    fseek(fp,0,SEEK_END);
    length=ftell(fp);
    fseek(fp,0,SEEK_SET);
    _sFort8=(char *)malloc(length);
    fread(_sFort8,length,1,fp);
    fclose(fp);

    return _sFort8;
}

void autoCallConv setFort8File(char* content, int length)
{
    string fileName = getFullPath("fort.8");

    if (_sFort8 != NULL)
        free (_sFort8);

    _sFort8 = strdup(content);
    _nFort8Length = length;

    ofstream f(fileName.c_str());
    f<<content;
    f.close();

//    FILE* fp = fopen(fileName, "wb");
//    fwrite(content, 1, length, fp);
//    fclose(fp);
}

char* autoCallConv getFort9File(int length)
{
    string fileName = getFullPath("fort.9");
    if (_sFort9 != NULL)
        free(_sFort9);

    length = 0;

    FILE* fp = fopen(fileName.c_str(), "rb");
    if (fp == NULL) return NULL;
    fseek(fp,0,SEEK_END);
    length=ftell(fp);
    fseek(fp,0,SEEK_SET);
    _sFort9=(char *)malloc(length);
    fread(_sFort9,length,1,fp);
    fclose(fp);

    return _sFort9;
}

void autoCallConv CallAuto(const string& tempFolder)
{
    char *oldDir = getcwd(NULL, 2048);
    try
    {
        int argc = 1;
        char cmd[512];
        strncpy(cmd, "vsAuto\0", 7);
        char *argv[] = { cmd }  ;
        chdir(tempFolder.c_str());
        autolib::AUTO_main(argc, argv);                
    }
    catch(exception& e)
    {
        //Re throw
        cerr<<"Exception in LibAuto: "<<e.what();

    }
    autolib::CloseAllFiles();
    clearCallbacks();
    chdir(oldDir);
    free(oldDir);
}

}

