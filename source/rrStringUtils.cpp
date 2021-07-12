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


namespace rr
{

char* createText(const std::string& str)
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

std::string replaceWord(const std::string& str1, const std::string& str2, const std::string& theString)
{
    std::string temp(theString);
    while(temp.find(str1) != std::string::npos)
    {
        temp.replace(temp.find(str1), str1.size(), str2);
    };

    return temp;
}

bool convertFunctionCallToUseVarArgsSyntax(const std::string& funcName, std::string& expression)
{
    size_t startFrom = expression.find(funcName);
    if(startFrom != std::string::npos)
    {
        //Convert this to variable syntax...
        size_t rightPos = findMatchingRightParenthesis(expression, startFrom);
        if(rightPos != std::string::npos)
        {
            std::string funcArgs = expression.substr(startFrom, rightPos - startFrom);
            int nrOfArgs    = getNumberOfFunctionArguments(funcArgs);

            //Convert to a va_list thing
            //insert nrOfArgs, jsut after leftPos
            expression.insert(startFrom + funcName.size() + 1, toString(nrOfArgs) + ", ");
        }
    }
    return true;
}

std::string removeChars(const std::string& str, const std::string& chars)
{
    std::string result(str);
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

size_t findMatchingRightParenthesis(const std::string& expression, const size_t startFrom)
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

int getNumberOfFunctionArguments(const std::string& expression)
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

std::string tabs(const int& nr)
{
    std::stringstream some_tabs;
    for(int i = 0;i < nr; i++)
    {
        some_tabs <<"\t";
    }
    return some_tabs.str();
}

std::string NL()
{
    std::stringstream newLine;
    newLine << std::endl;
    return newLine.str();
}


std::string getFileName(const std::string& fileN)
{
    std::string fName;
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

std::string getFileNameNoExtension(const std::string& fileN)
{
    std::string fName;
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

std::string getFileExtension(const std::string& fileN)
{
    std::string fExtension;

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

std::string getFilePath(const std::string& fileN)
{
    std::string path;
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

std::string changeFileExtensionTo(const std::string& _fName, const std::string& newExtension)
{
    //Be aware of the case
    //".\\fName"
    //where  the . is not part of the filename
    std::string path = getFilePath(_fName);
    std::string fName = getFileName(_fName);
    //First create the file name, remove current extension if it exists

    if(fName.find_last_of('.') != std::string::npos)
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

    return (std::filesystem::path(path) /= fName).string();
}

bool startsWith(const std::string& src, const std::string& sub)
{
    bool result = (src.compare(0, sub.size(), sub) == 0);
    return result;
}

bool endsWith(const std::string& src, const std::string& sub)
{
    bool result = (src.compare(src.size() - sub.size(), src.size(), sub) == 0);
    return result;
}

std::string trim(const std::string& str, const char& ch)
{
    std::string trimmed(str);
    std::string::size_type pos = trimmed.find_last_not_of(ch);
    if(pos != std::string::npos)
    {
        trimmed.erase(pos + 1);
        pos = trimmed.find_first_not_of(ch);
        if(pos != std::string::npos)
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

std::string& trim(std::string& str)
{
    return ltrim(rtrim(str));
}

std::string & ltrim(std::string & str)
{
  auto it2 =  std::find_if( str.begin() , str.end() , [](char ch){ return !std::isspace<char>(ch , std::locale::classic() ) ; } );
  str.erase( str.begin() , it2);
  return str;
}

std::string & rtrim(std::string & str)
{
  auto it1 =  std::find_if( str.rbegin() , str.rend() , [](char ch){ return !std::isspace<char>(ch , std::locale::classic() ) ; } );
  str.erase( it1.base() , str.end() );
  return str;
}

std::string removeNewLines(const std::string& str, const int& howMany)
{
    return substitute(str, "\n" , "", howMany);
}

std::string format(const std::string& src, const std::string& arg)
{
    return substitute(src, "{0}", arg);
}

std::string format(const std::string& src, const int& arg)
{
    return substitute(src, "{0}", toString(arg));
}

std::string format(const std::string& src, const std::string& arg1, const std::string& arg2)
{
    std::string tmp = substitute(src, "{0}", arg1);
    return substitute(tmp, "{1}", arg2);
}

std::string format(const std::string& src, const std::string& arg1, const int& arg2)
{
    std::string tmp = substitute(src, "{0}", arg1);
    return substitute(tmp, "{1}", toString(arg2));
}

std::string format(const std::string& src, const std::string& arg1, const std::string& arg2, const std::string& arg3)
{
    std::string tmp(src);
    tmp = substitute(tmp, "{0}", arg1);
    tmp = substitute(tmp, "{1}", arg2);
    tmp = substitute(tmp, "{2}", arg3);
    return tmp;
}

std::string format(const std::string& src, const std::string& arg1, const int& arg2, const std::string& arg3)
{
    std::string tmp = substitute(src, "{0}", arg1);
    tmp = substitute(tmp, "{1}", toString(arg2));
     return substitute(tmp, "{2}", arg3);
}

std::string format(const std::string& str1, const std::string& arg1, const std::string& arg2, const std::string& arg3, const std::string& arg4)
{
    std::string token1("{0}");
    std::string token2("{1}");
    std::string token3("{2}");
    std::string token4("{3}");
    std::string newString(str1);

    newString = substitute(newString, token1, arg1);
    newString = substitute(newString, token2, arg2);
    newString = substitute(newString, token3, arg3);
    newString = substitute(newString, token4, arg4);
    return newString;
}

std::string format(const std::string& str1, const std::string& arg1, const std::string& arg2, const std::string& arg3, const std::string& arg4, const std::string& arg5)
{
    std::string token1("{0}");
    std::string token2("{1}");
    std::string token3("{2}");
    std::string token4("{3}");
    std::string token5("{4}");
    std::string newString(str1);

    newString = substitute(newString, token1, arg1);
    newString = substitute(newString, token2, arg2);
    newString = substitute(newString, token3, arg3);
    newString = substitute(newString, token4, arg4);
    newString = substitute(newString, token5, arg5);
    return newString;
}

std::string format(const std::string& str1, const unsigned int& arg1, const std::string& arg2)
{
    std::string token1("{0}");
    std::string token2("{1}");
    std::string newString(str1);

    newString = substitute(newString, token1, (int) arg1);
    newString = substitute(newString, token2, arg2);
    return newString;
}

std::string format(const std::string& str1, const unsigned int& arg1, const std::string& arg2, const std::string& arg3)
{
    std::string token1("{0}");
    std::string token2("{1}");
    std::string token3("{2}");
    std::string newString(str1);

    newString = substitute(newString, token1, rr::toString(arg1));
    newString = substitute(newString, token2, arg2);
      newString = substitute(newString, token3, arg3);

    return newString;
}

std::string format(const std::string& str1, const unsigned int& arg1, const unsigned int& arg2, const std::string& arg3, const std::string& arg4)
{
    std::string tok1("{0}");
    std::string tok2("{1}");
    std::string tok3("{2}");
    std::string tok4("{2}");
    std::string newString(str1);

    newString = substitute(newString, tok1, arg1);
    newString = substitute(newString, tok2, arg2);
      newString = substitute(newString, tok3, arg3);
      newString = substitute(newString, tok4, arg4);

    return newString;
}

std::string substitute(const std::string& src, const std::string& thisOne, const int& withThisOne, const int& howMany)
{
    return substitute(src, thisOne, toString(withThisOne), howMany);
}

//std::string substitute(const std::string& src, const std::string& thisOne, const double& withThisOne, const int& howMany)
//{
//    return substitute(src, thisOne, toString(withThisOne), howMany);
//}

std::string substitute(const std::string& src, const std::string& thisOne, const std::string& withThisOne, const int& howMany)
{
    std::string newString(src);
    int count = 0;

    while(newString.find(thisOne) != std::string::npos)
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

//bool InStringList(const std::string& fldr, std::list<std::string>& theList)
//{
//    std::list<std::string>::iterator index = std::find_if(theList.begin(), theList.end(),  mtkCompareStrings(fldr));
//    return (index != theList.end()) ? true : false;
//}

std::string intToStr(const int& nt)
{
    //char *itoa(int value, char *std::string, int radix);
//    char str[100];
//    itoa(nt, str, 10);
    std::stringstream number;
    number<<nt;

//    std::string valStr(str);
    return number.str();
}

std::string dblToStr(const double& nt)
{
    char str[25];
    int sig = 5; /* significant digits */
    char* ptr =  gcvt(nt, sig, str);
    return std::string(str);
}

int strToInt(const std::string& str)
{
    return atoi(str.c_str());
}

double strToDbl(const std::string& str)
{
    char *endptr;
    return strtod(str.c_str(), &endptr);
}

std::vector<std::string> splitString(const std::string &text, const char& oneSep)
{
    std::string separator;
    separator = oneSep;
    return splitString(text, separator);
}

std::vector<std::string> splitString(const std::string &text, const std::string &separators)
{
    std::vector<std::string> words;
    size_t n = text.length();
    size_t start = text.find_first_not_of(separators);

    while( (start < n) )
    {
        size_t stop = text.find_first_of(separators, start);
        if( (stop == std::string::npos) || (stop > n) )
        {
            stop = n;
        }
        words.push_back(text.substr(start, stop - start));
        start = text.find_first_not_of(separators, stop+1);
    }

    return words;
}

std::vector<std::string> splitString(const std::string &text, const std::string &separators, bool cutDelimiter)
{
    std::vector<std::string> words;
    size_t n = text.length();
    size_t start = text.find_first_not_of(separators);
    while( (start != std::string::npos) && (start < n) )
    {
        size_t stop = text.find_first_of(separators, start);
        if( (stop == std::string::npos) || (stop > n) )
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

size_t splitString(std::vector<std::string>& words, const std::string &text, const std::string &separators)
{
    size_t n = text.length();
    size_t start = text.find_first_not_of(separators);
    while( (start != std::string::npos) && (start < n) )
    {
        size_t stop = text.find_first_of(separators, start);
        if( (stop == std::string::npos) || (stop > n) )
        {
            stop = n;
        }
        words.push_back(text.substr(start, stop - start));
        start = text.find_first_not_of(separators, stop+1);
    }

    return words.size();
}

int toInt(const std::string& str)
{
    return atoi(str.c_str());
}

bool toBool(const std::string& str)
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

double toDouble(const std::string& str)
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
        rrLog(Logger::LOG_WARNING) << "returning 0.0 for empty std::string in toDouble()";
        return 0;
    }

    std::string s = str;
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

    rrLog(Logger::LOG_WARNING) << "could not parse std::string \"" << str << "\" to double, returning NaN";
    return std::numeric_limits<double>::quiet_NaN();
}

std::vector<double> toDoubleVector(const std::string& str)
{
	// double std::vector is in the form "[d1,d2,d3...dn]"
	// get rid of '[' and ']' first

	size_t start = str.find("[");
	std::string st = str.substr(start + 1, str.size() - 1);
	// parse into std::string std::vector
	std::vector<std::string> parts(splitString(st, ","));
	std::vector<double> res;
	for (unsigned i = 0; i < parts.size(); i++)
		res.push_back(toDouble(parts[i]));
	return res;

}

complex<double> toComplex(const std::string& str)
{
    std::vector<std::string> parts(splitString(str,"(,)"));

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

std::string toUpperOrLowerCase(const std::string& inStr, int (*func)(int))
{
    std::string rString(inStr);
    std::transform(rString.begin(), rString.end(), rString.begin(), func);
    return rString;
}

std::string toUpper(const std::string& inStr)
{
    std::string rString(inStr);
    std::transform(rString.begin(), rString.end(), rString.begin(), (int(*)(int)) toupper);
    return rString;
}

std::string toLower(const std::string& inStr)
{
    std::string rString(inStr);
    std::transform(rString.begin(), rString.end(), rString.begin(), (int(*)(int)) tolower);
    return rString;
}

std::string toString(const char* str)
{
    return (std::string(str));
}

std::string toString(const bool& val)
{
    return val ? "true" : "false";
}

std::string toString(const double& val, const std::string& format)
{
    char sBuffer[256];
    sprintf(sBuffer, format.c_str(), val);
    return std::string(sBuffer);
}

//std::string toString(const int& val)
//{
//    char sBuffer[256];
//    sprintf(sBuffer, format.c_str(), val);
//    return std::string(sBuffer);
//}

std::string toString(const unsigned int& n, const std::string& format, const int nBase)
{
    char sBuffer[256];
    if (nBase == 16)
    {
        sprintf(sBuffer, "%X", n);
        return std::string("0x") + std::string(sBuffer);
    }
    else if(nBase == 2)
    {
        std::string tmp = "";
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
        return std::string(sBuffer);
    }
}

std::string toString(const int& n, const std::string& format, const int nBase)
{
    char sBuffer[256];
    if (nBase == 16)
    {
        sprintf(sBuffer, "%X", n);
        return std::string("0x") + std::string(sBuffer);
    }
    else if(nBase == 2)
    {
        std::string tmp = "";
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
        return std::string(sBuffer);
    }
}

std::string toString(const int n, const int nBase)
{
    char sBuffer[256];
    if (nBase == 16)
    {
        sprintf(sBuffer, "%X", n);
        return std::string("0x") + std::string(sBuffer);
    }
    else if(nBase == 2)
    {
        std::string tmp = "";
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
        return std::string(sBuffer);
    }
}

std::string toString(const long n, const int nBase)
{
    char sBuffer[256];
    if (nBase == 10)
    {
        sprintf(sBuffer, "%lu", n);
        return std::string(sBuffer);
    }
    return toString( int(n), nBase);
}

std::string toString(const unsigned long n, const int nBase)
{
    char sBuffer[256];
    if (nBase == 10)
    {
        sprintf(sBuffer, "%lu", n);
        return std::string(sBuffer);
    }
    return toString( int(n), nBase);
}

std::string toString(const unsigned short n, const int nBase)
{
    char sBuffer[256];
    if (nBase == 10)
    {
        sprintf(sBuffer, "%u", n);
        return std::string(sBuffer);
    }
    return toString( int(n), nBase);
}

std::string toString(const short n, const int nBase)
{
    return toString( int(n), nBase);
}

std::string toString(const char n)
{
    char sBuffer[256];
    sprintf(sBuffer, "%c", n);
    return std::string(sBuffer);
}

std::string toString(const unsigned char n)
{
    char sBuffer[256];
    sprintf(sBuffer, "%c", n);
    return std::string(sBuffer);
}

std::string toString(const std::string& str)
{
    return str;
}

std::string toString(const std::vector<std::string>& vec, const std::string& sep)
{
    std::stringstream text;
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

std::string toStringSize(size_t n)
{
    std::stringstream ss;
    ss << n;
    return ss.str();
}

int compareNoCase(const std::string& str1, const std::string& str2)
{
#if defined(_WIN32) || defined(__WIN32__)
    int res = stricmp(str1.c_str(), str2.c_str());
#else
    int res = strcasecmp(str1.c_str(), str2.c_str());
#endif
    return res;
}

std::string append(const std::string& str)
{
    //std::stringstream ss;
    //ss<<str;
    return str;
}

std::string append(const int& str)
{
    std::stringstream ss;
    ss<<str;
    return ss.str();
}


std::string append(const unsigned int& str)
{
    std::stringstream ss;
    ss<<str;
    return ss.str();
}


std::string append(const std::string& s1, const std::string& s2)
{
    std::stringstream ss;
    ss<<s1<<s2;
    return ss.str();
}

std::string append(const std::string& s1, const std::string& s2, const std::string& s3)
{
    std::stringstream ss;
    ss<<s1<<s2<<s3;
    return ss.str();
}

std::string append(const std::string& s1, const unsigned int& s2, const std::string& s3)
{
    std::stringstream ss;
    ss<<s1<<s2<<s3;
    return ss.str();
}

std::string append(const std::string& s1, const unsigned int& s2, const std::string& s3, const std::string& s4)
{
    std::stringstream ss;
    ss<<s1<<s2<<s3<<s4;
    return ss.str();
}

}

