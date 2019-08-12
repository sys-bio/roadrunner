#pragma hdrstop
#include <algorithm>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sstream>
#include <cstring>
#include <cctype>
#include <map>
#include "rrStringUtils.h"
#include "rrUtils.h"
#include "rrLogger.h"

using namespace std;
namespace rr
{

char* createText(const string& str)
{
    if(str.size() == 0)
    {
        return NULL;
    }

    char* text = (char*)malloc((str.size() + 1)*sizeof(char));
    std::copy(str.begin(), str.end(), text);
    text[str.size()] = '\0'; //terminating NULL!
    return text;
}

char* createText(const int& count)
{
    if(count == 0)
    {
        return NULL;
    }

    char* text = (char*)malloc((count + 1)*sizeof(char));
    text[count] = '\0'; //terminating NULL!
    return text;
}

bool freeText(char* str)
{
    free(str);
    return true;
}

string replaceWord(const string& str1, const string& str2, const string& theString)
{
    string temp(theString);
    while(temp.find(str1) != string::npos)
    {
        temp.replace(temp.find(str1), str1.size(), str2);
    };

    return temp;
}

bool convertFunctionCallToUseVarArgsSyntax(const string& funcName, string& expression)
{
    size_t startFrom = expression.find(funcName);
    if(startFrom != string::npos)
    {
        //Convert this to variable syntax...
        size_t rightPos = findMatchingRightParenthesis(expression, startFrom);
        if(rightPos != string::npos)
        {
            string funcArgs = expression.substr(startFrom, rightPos - startFrom);
            int nrOfArgs    = getNumberOfFunctionArguments(funcArgs);

            //Convert to a va_list thing
            //insert nrOfArgs, jsut after leftPos
            expression.insert(startFrom + funcName.size() + 1, toString(nrOfArgs) + ", ");
        }
    }
    return true;
}

string removeChars(const string& str, const string& chars)
{
    string result(str);
    for(int chrNr = 0; chrNr < chars.size(); chrNr++)
    {
           result.erase(std::remove(result.begin(), result.end(), chars[chrNr]), result.end());
    }

    return result;
}

bool isUnwantedChar(char ch) //Predicate for find_if algorithms..
{
    if(ch == '\n' || ch == '\t' || ch ==' ')
    {
        return true;
    }
    return false;
}

size_t findMatchingRightParenthesis(const string& expression, const size_t startFrom)
{
    int pCount = 0;

    size_t leftPos  = expression.find("(", startFrom);    //First lef parenthesis
    bool isScanning = false;    //Start scanning when first left parenthesis is found

    for(size_t i = startFrom; i < expression.size(); i++)
    {
        char ch = expression[i];
        if(ch == '(')
        {
            pCount++;
            isScanning = true;
        }
        if(ch == ')')
        {
               pCount--;
        }
        if(pCount == 0 && isScanning == true)
        {
            //found it..
            return i;
        }
    }

    return std::string::npos;
}

int getNumberOfFunctionArguments(const string& expression)
{
    int pCount = 0;    //count parenthesis
    int nrOfArgs = 1;
    bool isScanning = false;    //Start scanning when first left parenthesis is found
    for(int i = 0; i < expression.size(); i++)
    {
        char ch = expression[i];
        if(ch == '(')
        {
            isScanning = true;
            pCount++;
        }
        if(ch == ')')
        {
               pCount--;
        }
        if(ch == ',' && pCount == 1 && isScanning == true)
        {
            nrOfArgs++;
        }
     }

     if(expression.size() == 0)
     {
         return -1;
     }
     return nrOfArgs;
}

string tabs(const int& nr)
{
    stringstream some_tabs;
    for(int i = 0;i < nr; i++)
    {
        some_tabs <<"\t";
    }
    return some_tabs.str();
}

string NL()
{
    stringstream newLine;
    newLine << endl;
    return newLine.str();
}


string getFileName(const string& fileN)
{
    string fName;
    if(fileN.find_last_of( '\\' ) != std::string::npos)
    {
        fName = fileN.substr(fileN.find_last_of( '\\' )+ 1, fileN.size());
        return fName;
    }
    else if(fileN.find_last_of( '/' ) != std::string::npos)
    {
        fName = fileN.substr(fileN.find_last_of( '/' ) + 1, fileN.size());
        return fName;
    }

    return fileN; //There was no path in present..
}

string getFileNameNoExtension(const string& fileN)
{
    string fName;
    if(fileN.find_last_of( '\\' ) != std::string::npos)
    {
        fName = fileN.substr(fileN.find_last_of( '\\' )+ 1, fileN.size());
    }
    else if(fileN.find_last_of( '/' ) != std::string::npos)
    {
        fName = fileN.substr(fileN.find_last_of( '/' ) + 1, fileN.size());
    }
    else
    {
        fName = fileN;
    }

    return changeFileExtensionTo(fName, "");
}

string getFileExtension(const string& fileN)
{
    string fExtension;

    fExtension = getFileName(fileN);

    if(fileN.find_last_of( '.' ) != std::string::npos)
    {
        fExtension = fileN.substr(fileN.find_last_of('.')+ 1);
        return fExtension;
    }
    else
    {
        return "";
    }
}

string getFilePath(const string& fileN)
{
    string path;
    if(fileN.find_last_of( '\\' ) != std::string::npos)
    {
        path = fileN.substr( 0, fileN.find_last_of( '\\' ));
        return path;
    }
    else if(fileN.find_last_of( '/' ) != std::string::npos)
    {
        path = fileN.substr( 0, fileN.find_last_of( '/' ));
        return path;
    }

    return "";
}

string changeFileExtensionTo(const string& _fName, const string& newExtension)
{
    //Be aware of the case
    //".\\fName"
    //where  the . is not part of the filename
    string path = getFilePath(_fName);
    string fName = getFileName(_fName);
    //First create the file name, remove current extension if it exists

    if(fName.find_last_of('.') != string::npos)
    {
        //Extension does exist. Cut it, and append new one
        fName =  fName.substr(0, fName.find_last_of('.'));
    }

    if(newExtension[0] == '.')
    {
        fName = fName + newExtension;
    }
    else if(newExtension.size() == 0)    //No extension
    {
        return fName;
    }
    else
    {
        fName = fName + "." + newExtension;
    }

    return joinPath(path, fName);
}

bool startsWith(const string& src, const string& sub)
{
    bool result = (src.compare(0, sub.size(), sub) == 0);
    return result;
}

bool endsWith(const string& src, const string& sub)
{
    bool result = (src.compare(src.size() - sub.size(), src.size(), sub) == 0);
    return result;
}

string trim(const string& str, const char& ch)
{
    string trimmed(str);
    string::size_type pos = trimmed.find_last_not_of(ch);
    if(pos != string::npos)
    {
        trimmed.erase(pos + 1);
        pos = trimmed.find_first_not_of(ch);
        if(pos != string::npos)
        {
            trimmed.erase(0, pos);
        }
      }
    else
    {
        trimmed.erase(trimmed.begin(), trimmed.end());
    }
    return trimmed;
}

string removeNewLines(const string& str, const int& howMany)
{
    return substitute(str, "\n" , "", howMany);
}

string format(const string& src, const string& arg)
{
    return substitute(src, "{0}", arg);
}

string format(const string& src, const int& arg)
{
    return substitute(src, "{0}", toString(arg));
}

string format(const string& src, const string& arg1, const string& arg2)
{
    string tmp = substitute(src, "{0}", arg1);
    return substitute(tmp, "{1}", arg2);
}

string format(const string& src, const string& arg1, const int& arg2)
{
    string tmp = substitute(src, "{0}", arg1);
    return substitute(tmp, "{1}", toString(arg2));
}

string format(const string& src, const string& arg1, const string& arg2, const string& arg3)
{
    string tmp(src);
    tmp = substitute(tmp, "{0}", arg1);
    tmp = substitute(tmp, "{1}", arg2);
    tmp = substitute(tmp, "{2}", arg3);
    return tmp;
}

string format(const string& src, const string& arg1, const int& arg2, const string& arg3)
{
    string tmp = substitute(src, "{0}", arg1);
    tmp = substitute(tmp, "{1}", toString(arg2));
     return substitute(tmp, "{2}", arg3);
}

string format(const string& str1, const string& arg1, const string& arg2, const string& arg3, const string& arg4)
{
    string token1("{0}");
    string token2("{1}");
    string token3("{2}");
    string token4("{3}");
    string newString(str1);

    newString = substitute(newString, token1, arg1);
    newString = substitute(newString, token2, arg2);
    newString = substitute(newString, token3, arg3);
    newString = substitute(newString, token4, arg4);
    return newString;
}

string format(const string& str1, const string& arg1, const string& arg2, const string& arg3, const string& arg4, const string& arg5)
{
    string token1("{0}");
    string token2("{1}");
    string token3("{2}");
    string token4("{3}");
    string token5("{4}");
    string newString(str1);

    newString = substitute(newString, token1, arg1);
    newString = substitute(newString, token2, arg2);
    newString = substitute(newString, token3, arg3);
    newString = substitute(newString, token4, arg4);
    newString = substitute(newString, token5, arg5);
    return newString;
}

string format(const string& str1, const unsigned int& arg1, const string& arg2)
{
    string token1("{0}");
    string token2("{1}");
    string newString(str1);

    newString = substitute(newString, token1, (int) arg1);
    newString = substitute(newString, token2, arg2);
    return newString;
}

string format(const string& str1, const unsigned int& arg1, const string& arg2, const string& arg3)
{
    string token1("{0}");
    string token2("{1}");
    string token3("{2}");
    string newString(str1);

    newString = substitute(newString, token1, rr::toString(arg1));
    newString = substitute(newString, token2, arg2);
      newString = substitute(newString, token3, arg3);

    return newString;
}

string format(const string& str1, const unsigned int& arg1, const unsigned int& arg2, const string& arg3, const string& arg4)
{
    string tok1("{0}");
    string tok2("{1}");
    string tok3("{2}");
    string tok4("{2}");
    string newString(str1);

    newString = substitute(newString, tok1, arg1);
    newString = substitute(newString, tok2, arg2);
      newString = substitute(newString, tok3, arg3);
      newString = substitute(newString, tok4, arg4);

    return newString;
}

string substitute(const string& src, const string& thisOne, const int& withThisOne, const int& howMany)
{
    return substitute(src, thisOne, toString(withThisOne), howMany);
}

//string substitute(const string& src, const string& thisOne, const double& withThisOne, const int& howMany)
//{
//    return substitute(src, thisOne, toString(withThisOne), howMany);
//}

string substitute(const string& src, const string& thisOne, const string& withThisOne, const int& howMany)
{
    string newString(src);
    int count = 0;

    while(newString.find(thisOne) != string::npos)
    {
        if(count == howMany)
        {
            break;
        }
        else
        {
            newString.replace(newString.find(thisOne), thisOne.size(), withThisOne);
            count++;
        }
    }
    return newString;
}

//bool InStringList(const string& fldr, list<string>& theList)
//{
//    list<string>::iterator index = std::find_if(theList.begin(), theList.end(),  mtkCompareStrings(fldr));
//    return (index != theList.end()) ? true : false;
//}

string intToStr(const int& nt)
{
    //char *itoa(int value, char *string, int radix);
//    char str[100];
//    itoa(nt, str, 10);
    stringstream number;
    number<<nt;

//    string valStr(str);
    return number.str();
}

string dblToStr(const double& nt)
{
    char str[25];
    int sig = 5; /* significant digits */
    char* ptr =  gcvt(nt, sig, str);
    return string(str);
}

int strToInt(const string& str)
{
    return atoi(str.c_str());
}

double strToDbl(const string& str)
{
    char *endptr;
    return strtod(str.c_str(), &endptr);
}

vector<string> splitString(const string &text, const char& oneSep)
{
    string separator;
    separator = oneSep;
    return splitString(text, separator);
}

vector<string> splitString(const string &text, const string &separators)
{
    vector<string> words;
    int n = text.length();
    int start = text.find_first_not_of(separators);

    while( (start >= 0) && (start < n) )
    {
        int stop = text.find_first_of(separators, start);
        if( (stop < 0) || (stop > n) )
        {
            stop = n;
        }
        words.push_back(text.substr(start, stop - start));
        start = text.find_first_not_of(separators, stop+1);
    }

    return words;
}

vector<string> splitString(const string &text, const string &separators, bool cutDelimiter)
{
    vector<string> words;
    int n = text.length();
    int start = text.find_first_not_of(separators);
    while( (start >= 0) && (start < n) )
    {
        int stop = text.find_first_of(separators, start);
        if( (stop < 0) || (stop > n) )
        {
            stop = n;
        }

        if(cutDelimiter)
        {
            words.push_back(text.substr(start, stop - start));
        }
        else
        {
            start -= 1;
            if(stop != n)
            {
                stop += 1;
            }
            words.push_back(text.substr(start, stop - start));
        }
        start = text.find_first_not_of(separators, stop+1);
    }

    return words;
}

int splitString(vector<string>& words, const string &text, const string &separators)
{
    int n = text.length();
    int start = text.find_first_not_of(separators);
    while( (start >= 0) && (start < n) )
    {
        int stop = text.find_first_of(separators, start);
        if( (stop < 0) || (stop > n) )
        {
            stop = n;
        }
        words.push_back(text.substr(start, stop - start));
        start = text.find_first_not_of(separators, stop+1);
    }

    return words.size();
}

int toInt(const string& str)
{
    return atoi(str.c_str());
}

bool toBool(const string& str)
{
    if(str.size() < 2)
    {
        return (str == "1")     ? true : false;
    }
    else
    {
        //Could have a case insensitive comparison..
        return (str == "true") || (str == "True") || (str == "TRUE") ? true : false;
    }
}

double toDouble(const string& str)
{
    // The MSVC strtod is buggy in that it does not parse InF and NaN values,
    // so have to compensate.
    // go ahead and have the same behavior on all platforms so its easier
    // to diagnose issues.
    const char* cstr = str.c_str();
    char* endptr = const_cast<char*>(cstr);
    double result = ::strtod(cstr, &endptr);

    if (endptr > cstr)
    {
        return result;
    }

    if(str.length() == 0)
    {
        return 0;
    }

    if(!str.size())
    {
        Log(Logger::LOG_WARNING) << "returning 0.0 for empty string in toDouble()";
        return 0;
    }

    string s = str;
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);

    if(s.find("NAN") != std::string::npos || s == "-")
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    size_t inf = s.find("INF");

    if(inf != std::string::npos)
    {
        size_t negpos = s.find("-");
        if(negpos != std::string::npos && negpos < inf) // -inf
        {
            return -std::numeric_limits<double>::infinity();
        }
        else // +inf
        {
            return std::numeric_limits<double>::infinity();
        }
    }

    Log(Logger::LOG_WARNING) << "could not parse string \"" << str << "\" to double, returning NaN";
    return std::numeric_limits<double>::quiet_NaN();
}

vector<double> toDoubleVector(const string& str)
{
	// double vector is in the form "[d1,d2,d3...dn]"
	// get rid of '[' and ']' first

	size_t start = str.find("[");
	string st = str.substr(start + 1, str.size() - 1);
	// parse into string vector
	vector<string> parts(splitString(st, ","));
	vector<double> res;
	for (unsigned i = 0; i < parts.size(); i++)
		res.push_back(toDouble(parts[i]));
	return res;

}

complex<double> toComplex(const string& str)
{
    vector<string> parts(splitString(str,"(,)"));

    if(parts.size() != 2)
    {
        //should throw...
        return complex<double>(0,0);
    }

    char *endptr = NULL;
    complex<double> num;
    if(parts[0] == "-")
    {
        if(parts[1] == "-")
        {
            return complex<double>(gDoubleNaN, gDoubleNaN);
        }
        double im = strtod(parts[1].c_str(), &endptr);
        return complex<double>(gDoubleNaN,im);
    }

    if(parts[1] == "-")
    {
        double re = strtod(parts[0].c_str(), &endptr);
        return complex<double>(re,gDoubleNaN);
    }

    double re = strtod(parts[0].c_str(), &endptr);
    double im = strtod(parts[1].c_str(), &endptr);

    return complex<double>(re,im);
}

string toUpperOrLowerCase(const string& inStr, int (*func)(int))
{
    string rString(inStr);
    std::transform(rString.begin(), rString.end(), rString.begin(), func);
    return rString;
}

string toUpper(const string& inStr)
{
    string rString(inStr);
    std::transform(rString.begin(), rString.end(), rString.begin(), (int(*)(int)) toupper);
    return rString;
}

string toLower(const string& inStr)
{
    string rString(inStr);
    std::transform(rString.begin(), rString.end(), rString.begin(), (int(*)(int)) tolower);
    return rString;
}

string toString(const char* str)
{
    return (string(str));
}

string toString(const bool& val)
{
    return val ? "true" : "false";
}

string toString(const double& val, const string& format)
{
    char sBuffer[256];
    sprintf(sBuffer, format.c_str(), val);
    return string(sBuffer);
}

//string toString(const int& val)
//{
//    char sBuffer[256];
//    sprintf(sBuffer, format.c_str(), val);
//    return string(sBuffer);
//}

string toString(const unsigned int& n, const string& format, const int nBase)
{
    char sBuffer[256];
    if (nBase == 16)
    {
        sprintf(sBuffer, "%X", n);
        return string("0x") + string(sBuffer);
    }
    else if(nBase == 2)
    {
        string tmp = "";
        int k = n;
        for (int i=0; i<8; i++)
        {
            if ((k & 0x80) != 0)
                tmp += "1";
            else
                tmp += "0";
            k = k<<1;
        }
        return "0b" + tmp;
    }
    else
    {
        sprintf(sBuffer, "%d", n);
        return string(sBuffer);
    }
}

string toString(const int& n, const string& format, const int nBase)
{
    char sBuffer[256];
    if (nBase == 16)
    {
        sprintf(sBuffer, "%X", n);
        return string("0x") + string(sBuffer);
    }
    else if(nBase == 2)
    {
        string tmp = "";
        int k = n;
        for (int i=0; i<8; i++)
        {
            if ((k & 0x80) != 0)
                tmp += "1";
            else
                tmp += "0";
            k = k<<1;
        }
        return "0b" + tmp;
    }
    else
    {
        sprintf(sBuffer, "%d", n);
        return string(sBuffer);
    }
}

string toString(const int n, const int nBase)
{
    char sBuffer[256];
    if (nBase == 16)
    {
        sprintf(sBuffer, "%X", n);
        return string("0x") + string(sBuffer);
    }
    else if(nBase == 2)
    {
        string tmp = "";
        int k = n;
        for (int i=0; i<8; i++)
        {
            if ((k & 0x80) != 0)
                tmp += "1";
            else
                tmp += "0";
            k = k<<1;
        }
        return "0b" + tmp;
    }
    else
    {
        sprintf(sBuffer, "%d", n);
        return string(sBuffer);
    }
}

string toString(const long n, const int nBase)
{
    char sBuffer[256];
    if (nBase == 10)
    {
        sprintf(sBuffer, "%lu", n);
        return string(sBuffer);
    }
    return toString( int(n), nBase);
}

string toString(const unsigned long n, const int nBase)
{
    char sBuffer[256];
    if (nBase == 10)
    {
        sprintf(sBuffer, "%lu", n);
        return string(sBuffer);
    }
    return toString( int(n), nBase);
}

string toString(const unsigned short n, const int nBase)
{
    char sBuffer[256];
    if (nBase == 10)
    {
        sprintf(sBuffer, "%u", n);
        return string(sBuffer);
    }
    return toString( int(n), nBase);
}

string toString(const short n, const int nBase)
{
    return toString( int(n), nBase);
}

string toString(const char n)
{
    char sBuffer[256];
    sprintf(sBuffer, "%c", n);
    return string(sBuffer);
}

string toString(const unsigned char n)
{
    char sBuffer[256];
    sprintf(sBuffer, "%c", n);
    return string(sBuffer);
}

string toString(const string& str)
{
    return str;
}

string toString(const vector<string>& vec, const string& sep)
{
    stringstream text;
    text<<"{";
    for(int i = 0; i < vec.size(); i++)
    {
        text<<vec[i];
        if(i < vec.size() - 1)
        {
            text<<sep;
        }
    }
    text<<"}";
    return text.str();
}

int compareNoCase(const string& str1, const string& str2)
{
#if defined(_WIN32) || defined(__WIN32__)
    int res = stricmp(str1.c_str(), str2.c_str());
#else
    int res = strcasecmp(str1.c_str(), str2.c_str());
#endif
    return res;
}

string append(const string& str)
{
    //stringstream ss;
    //ss<<str;
    return str;
}

string append(const int& str)
{
    stringstream ss;
    ss<<str;
    return ss.str();
}


string append(const unsigned int& str)
{
    stringstream ss;
    ss<<str;
    return ss.str();
}


string append(const string& s1, const string& s2)
{
    stringstream ss;
    ss<<s1<<s2;
    return ss.str();
}

string append(const string& s1, const string& s2, const string& s3)
{
    stringstream ss;
    ss<<s1<<s2<<s3;
    return ss.str();
}

string append(const string& s1, const unsigned int& s2, const string& s3)
{
    stringstream ss;
    ss<<s1<<s2<<s3;
    return ss.str();
}

string append(const string& s1, const unsigned int& s2, const string& s3, const string& s4)
{
    stringstream ss;
    ss<<s1<<s2<<s3<<s4;
    return ss.str();
}

}

