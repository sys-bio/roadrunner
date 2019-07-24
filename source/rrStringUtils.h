#ifndef rrStringUtilsH
#define rrStringUtilsH
#include <string>
#include <list>
#include <vector>
#include <complex>
#include "rrConstants.h"
#include "rrExporter.h"
#include "rr-libstruct/lsMatrix.h"

namespace rr
{
using std::string;
using std::list;
using std::vector;
using std::complex;

RR_DECLSPEC char*               createText(const string& str);
RR_DECLSPEC char*               createText(const int& count);
RR_DECLSPEC bool                   freeText(char* str);

RR_DECLSPEC unsigned int        indexOf(const string& text, char checkFor);

RR_DECLSPEC double              extractDouble(std::string const& s, bool failIfLeftoverChars = false);

RR_DECLSPEC string              replaceWord(const string& str1, const string& str2, const string& theString);
RR_DECLSPEC bool                convertFunctionCallToUseVarArgsSyntax(const string& funcName, string& expression);
RR_DECLSPEC string              removeChars(const string& str, const string& chars);
RR_DECLSPEC bool                isUnwantedChar(char ch); //Predicate for find_if algorithms..
RR_DECLSPEC size_t              findMatchingRightParenthesis(const string& expression, const size_t startFrom);
RR_DECLSPEC int                 getNumberOfFunctionArguments(const string& expression);
RR_DECLSPEC string              tabs(const int& nr);
RR_DECLSPEC string              NL();

RR_DECLSPEC string                toUpperOrLowerCase(const string& inStr, int (*func)(int));
RR_DECLSPEC string                 toUpper(const string& str);
RR_DECLSPEC string                 toLower(const string& str);

RR_DECLSPEC string              getFilePath(const string& fileN);
RR_DECLSPEC string              getFileName(const string& fileN);
RR_DECLSPEC string              getFileNameNoExtension(const string& fileN);
RR_DECLSPEC string              getFileExtension(const string& fileN);

RR_DECLSPEC string              changeFileExtensionTo(const string& theFileName, const string& newExtension);

RR_DECLSPEC int                 compareNoCase(const string& str1, const string& str2);
RR_DECLSPEC string              trim(const string& str, const char& toTrim = ' ');
RR_DECLSPEC bool                startsWith(const string& src, const string& sub);
RR_DECLSPEC bool                endsWith(const string& src, const string& sub);

//conversions
RR_DECLSPEC string              intToStr(const int& nt);
RR_DECLSPEC int                 strToInt(const string& nt);
RR_DECLSPEC string              dblToStr(const double& nt);
RR_DECLSPEC double              strToDbl(const string& nt);
RR_DECLSPEC vector<string>      splitString(const string &text, const string &separators);
RR_DECLSPEC vector<string>      splitString(const string& input, const char& delimiters);
RR_DECLSPEC int                 toInt(const string& str);
RR_DECLSPEC bool                toBool(const string& str);
RR_DECLSPEC double              toDouble(const string& str);
RR_DECLSPEC complex<double>     toComplex(const string& str);

RR_DECLSPEC string              toString(const bool& b);
RR_DECLSPEC string              toString(const double& d, const string& format = gDoubleFormat);
RR_DECLSPEC string              toString(const unsigned int& n, const string& format = gIntFormat, const int nBase=10);
RR_DECLSPEC string              toString(const int& n, const string& format = gIntFormat, const int nBase=10);
RR_DECLSPEC string              toString(const long n, const int nBase=10);
RR_DECLSPEC string              toString(const unsigned long n, const int nBase=10);
RR_DECLSPEC string              toString(const unsigned short n, const int nBase=10);
RR_DECLSPEC string              toString(const short n, const int nBase=10);
RR_DECLSPEC string              toString(const char n);
RR_DECLSPEC string              toString(const unsigned char n);
RR_DECLSPEC string              toString(const string& s);
RR_DECLSPEC string              toString(const char* str);
RR_DECLSPEC string              toString(const vector<string>& vec, const string& sep = ", ");
RR_DECLSPEC string              toString(const vector<int>& vec, const string& sep = ", ");
RR_DECLSPEC string              toString(const vector<double>& vec, const string& sep = ", ");
//RR_DECLSPEC string              toString(const ls::Matrix<double> mat);
RR_DECLSPEC string              toString(const ls::Matrix<double>& mat);

RR_DECLSPEC string              format(const string& src, const int& arg);
RR_DECLSPEC string              format(const string& str, const int& arg1);


RR_DECLSPEC string              format(const string& src, const string& arg);
RR_DECLSPEC string              format(const string& src, const string& arg1, const string& arg2, const string& arg3);
RR_DECLSPEC string              format(const string& src, const string& arg1, const string& arg2);
RR_DECLSPEC string              format(const string& src, const string& arg1, const int& arg2);
RR_DECLSPEC string              format(const string& src, const string& arg1, const int& arg2, const string& arg3);
RR_DECLSPEC string              format(const string& str1, const string& str2);
RR_DECLSPEC string              format(const string& str1, const string& arg1, const string& arg2);
RR_DECLSPEC string              format(const string& str1, const string& arg1, const int& arg2);
RR_DECLSPEC string              format(const string& str1, const string& arg1, const int& arg2, const string& arg3);
RR_DECLSPEC string              format(const string& str1, const string& arg1, const string& arg2, const string& arg3);
RR_DECLSPEC string              format(const string& str1, const string& arg1, const string& arg2, const string& arg3, const string& arg4);
RR_DECLSPEC string              format(const string& str1, const string& arg1, const string& arg2, const string& arg3, const string& arg4, const string& arg5);

RR_DECLSPEC string              format(const string& str1, const unsigned int& arg1, const string& arg2);

RR_DECLSPEC string              format(const string& str1, const unsigned int& arg1, const string& arg2, const string& arg3);

RR_DECLSPEC string              format(const string& str1, const unsigned int& arg1, const unsigned int& arg2, const string& arg3, const string& arg4);

RR_DECLSPEC string              append(const string& str);
RR_DECLSPEC string              append(const int& str);
RR_DECLSPEC string              append(const unsigned int& str);
RR_DECLSPEC string              append(const string& s1, const string& s2);
RR_DECLSPEC string              append(const string& s1, const string& s2, const string& s3);
RR_DECLSPEC string              append(const string& s1, const unsigned int& s2, const string& s3);
RR_DECLSPEC string              append(const string& s1, const unsigned int& s2, const string& s3, const string& s4);

RR_DECLSPEC string              substitute(const string& src, const string& thisOne, const string& withThisOne, const int& howMany = -1);
RR_DECLSPEC string              substitute(const string& src, const string& thisOne, const int& withThisOne, const int& howMany = -1);
//RR_DECLSPEC string              substitute(const string& src, const string& thisOne, const double& withThisOne, const int& howMany = -1);
RR_DECLSPEC string              removeNewLines(const string& str, const int& howMany = -1);

/**
    \brief Template function, substituting an occurence of a string, target, in the source string, with another string, item. The howMany argument
    limits the number of substitutions. A value of -1 causes all occurences to be substituted. 
    The new string with the substitions is returned.
*/
template <class T>
inline string substituteN(const string& source, const string& target, const T& item, const int& howMany= -1)
{
    return substitute(source, target, toString(item), howMany);
}

/**
    \brief Template function, substituting an occurence of a string, target, within the source string, with another, item. The howMany argument
    limits the number of substitutions. A value of -1 causes all occurences to be substituted.
    The new string with the substitions is returned.
*/
template<>
inline string substituteN<double>(const string& source, const string& target, const double& item, const int& howMany)
{
    return substitute(source, target, toString(item, "%G"), howMany);
}

/**
    \brief Template format function. A "{0}" occurence in the src string is substituted with the
    value in arg1. The new string with the substition is returned.
*/
template <class T>
string formatN(const string& src, const T& arg1)
{
    string newString(src);
    string tok1("{0}");
    newString = substituteN(newString, tok1, arg1);
    return newString;
}

/**
    \brief Template format function. A "{i}" occurence in the src string, where i = 0 or 1, is substituted with the
    values in arg1 and arg2, respectively. The new string with the substitions is returned.
*/
template <class A, class B>
string formatN(const string& src, const A& arg1, const B& arg2)
{
    string newString(src);
    string tok1("{0}"), tok2("{1}");

    newString = substituteN(newString, tok1, arg1);
    newString = substituteN(newString, tok2, arg2);
    return newString;
}

/**
    \brief Template format function. A "{i}" occurence in the src string, where i = 0, 1 or 2, is substituted with the
    values in arg1, arg2 and arg3, respectively. The new string with the substitions is returned.
*/
template <class A, class B, class C>
string formatN(const string& src, const A& arg1, const B& arg2, const C& arg3)
{
    string newString(src);
    string tok1("{0}"), tok2("{1}"), tok3("{2}");

    newString = substituteN(newString, tok1, arg1, -1);
    newString = substituteN(newString, tok2, arg2, -1);
    newString = substituteN(newString, tok3, arg3, -1);
    return newString;
}

/**
    \brief Template format function. A "{i}" occurence in the src string, where i = 0, 1, 2 or 3, is substituted with the 
    values in arg1.. trough arg4, respectively. The new string with the substitions is returned.
*/
template <class A, class B, class C, class D>
string formatN(const string& src, const A& arg1, const B& arg2, const C& arg3, const D& arg4)
{
    string newString(src);
    string  tok1("{0}"), tok2("{1}"), tok3("{2}"),
            tok4("{3}");

    newString = substituteN(newString, tok1, arg1, -1);
    newString = substituteN(newString, tok2, arg2, -1);
    newString = substituteN(newString, tok3, arg3, -1);
    newString = substituteN(newString, tok4, arg4, -1);
    return newString;
}

/**
    \brief Template format function. A "{i}" occurence in the src string, where i = 0, 1, ..., 4, is substituted with the 
    values in arg1.. trough arg5, respectively. The new string with the substitions is returned.
*/
template <class A, class B, class C, class D, class E>
string formatN(const string& src, const A& arg1, const B& arg2, const C& arg3, const D& arg4, const E& arg5)
{
    string newString(src);
    string  tok1("{0}"), tok2("{1}"), tok3("{2}"),
            tok4("{3}"), tok5("{4}");

    newString = substituteN(newString, tok1, arg1);
    newString = substituteN(newString, tok2, arg2, -1);
    newString = substituteN(newString, tok3, arg3, -1);
    newString = substituteN(newString, tok4, arg4, -1);
    newString = substituteN(newString, tok5, arg5, -1);
    return newString;
}

/**
    \brief Template format function. A "{i}" occurence in the src string, where i = 0, 1, ..., 5, is substituted with the 
    values in arg1 trough arg5, respectively. The new string with the substitions is returned.
*/
template <class A, class B, class C, class D, class E, class F>
string formatN(const string& src, const A& arg1, const B& arg2, const C& arg3, const D& arg4, const E& arg5, const F& arg6)
{
    string newString(src);
    string  tok1("{0}"), tok2("{1}"), tok3("{2}"),
            tok4("{3}"), tok5("{4}"), tok6("{5}");

    newString = substituteN(newString, tok1, arg1, -1);
    newString = substituteN(newString, tok2, arg2, -1);
    newString = substituteN(newString, tok3, arg3, -1);
    newString = substituteN(newString, tok4, arg4, -1);
    newString = substituteN(newString, tok5, arg5, -1);
    newString = substituteN(newString, tok6, arg6, -1);

    return newString;
}

/**
    \brief Template format function. A "{i}" occurence in the src string, where i = 0, 1, ..., 6, is substituted with the 
    values in arg1 trough arg7, respectively. The new string with the substitions is returned.
*/
template <class A, class B, class C, class D, class E, class F, class G>
string formatN(const string& src, const A& arg1, const B& arg2, const C& arg3, const D& arg4, const E& arg5, const F& arg6, const G& arg7)
{
    string newString(src);
    string  tok1("{0}"), tok2("{1}"), tok3("{2}"),
            tok4("{3}"), tok5("{4}"), tok6("{5}"),
            tok7("{6}");

    newString = substituteN(newString, tok1, arg1, -1);
    newString = substituteN(newString, tok2, arg2, -1);
    newString = substituteN(newString, tok3, arg3, -1);
    newString = substituteN(newString, tok4, arg4, -1);
    newString = substituteN(newString, tok5, arg5, -1);
    newString = substituteN(newString, tok6, arg6, -1);
    newString = substituteN(newString, tok7, arg7, -1);

    return newString;
}

/**
    \brief Template format function. A "{i}" occurence in the src string, where i = 0, 1, ..., 7, is substituted with the 
    values in arg1 trough arg8, respectively. The new string with the substitions is returned.
*/
template <class A, class B, class C, class D, class E, class F, class G, class H>
string formatN(const string& src, const A& arg1, const B& arg2, const C& arg3, const D& arg4, const E& arg5, const F& arg6, const G& arg7, const H& arg8)
{
    string newString(src);
    string  tok1("{0}"), tok2("{1}"), tok3("{2}"),
            tok4("{3}"), tok5("{4}"), tok6("{5}"),
            tok7("{6}"), tok8("{7}");

    newString = substituteN(newString, tok1, arg1, -1);
    newString = substituteN(newString, tok2, arg2, -1);
    newString = substituteN(newString, tok3, arg3, -1);
    newString = substituteN(newString, tok4, arg4, -1);
    newString = substituteN(newString, tok5, arg5, -1);
    newString = substituteN(newString, tok6, arg6, -1);
    newString = substituteN(newString, tok7, arg7, -1);
    newString = substituteN(newString, tok8, arg8, -1);

    return newString;
}


}
#endif
