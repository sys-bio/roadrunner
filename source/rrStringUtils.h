#ifndef rrStringUtilsH
#define rrStringUtilsH
#include <string>
#include <list>
#include <vector>
#include <complex>
#include <map>
#include <unordered_map>
#include <set>
#include "rrConstants.h"
#include "rrExporter.h"

namespace rr
{
using std::string;
using std::list;
using std::vector;
using std::complex;

RR_DECLSPEC char*               createText(const string& str);
RR_DECLSPEC char*               createText(const int& count);
RR_DECLSPEC bool                   freeText(char* str);
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



template <typename T>
inline void                saveBinary(std::ostream& out, const T& t)
{
	out.write((char*)&t, sizeof(T));
}

template <>
inline void                saveBinary(std::ostream& out, const std::string& s)
{
	saveBinary(out, s.size());
	out.write(s.c_str(), s.size());
}

template <typename T>
inline void                saveBinary(std::ostream& out, const std::vector<T>& v)
{
	saveBinary(out, v.size());
	for (T t : v)
	{
		saveBinary(out, t);
	}
}

template <typename K, typename V>
inline void saveBinary(std::ostream& out, const std::map<K, V>& m)
{
	saveBinary(out, m.size());
	for (std::pair<K, V> p : m)
	{
		saveBinary(out, p.first);
		saveBinary(out, p.second);
	}
}

template <typename K, typename V>
inline void saveBinary(std::ostream& out, const std::unordered_map<K, V>& m)
{
	saveBinary(out, m.size());
	for (std::pair<K, V> p : m)
	{
		saveBinary(out, p.first);
		saveBinary(out, p.second);
	}
}

template <typename T>
inline void                saveBinary(std::ostream& out, const std::set<T>& s)
{
	saveBinary(out, s.size());
	for (T t : s)
	{
		saveBinary(out, t);
	}
}

template <typename T>
inline void                loadBinary(std::istream& in, T& t)
{
	T temp;
	in.read((char*)&temp, sizeof(T));
	t = temp;
}

template <>
inline void                loadBinary(std::istream& in, std::string& s)
{
	size_t ssize;
	loadBinary(in, ssize);
	s.resize(ssize);
	in.read(&s[0], ssize);
}

template <typename T>
inline void                loadBinary(std::istream& in, std::vector<T>& v)
{
	size_t vsize;
	loadBinary(in, vsize);
	v.clear();
	for (int i = 0; i < vsize; i++)
	{
		T next;
		loadBinary(in, next);
		v.push_back(next);
	}
}

template <typename K, typename V>
inline void loadBinary(std::istream& in, std::map<K, V>& m)
{
	size_t msize;
	loadBinary(in, msize);
	//m.clear();
	for (int i = 0; i < msize; i++)
	{
		K k;
		V v;
		loadBinary(in, k);
		loadBinary(in, v);
		m.emplace(k, v);
	}
}

template <typename K, typename V>
inline void loadBinary(std::istream& in, std::unordered_map<K, V>& m)
{
	size_t msize;
	loadBinary(in, msize);
	m.clear();
	for (int i = 0; i < msize; i++)
	{
		K k;
		V v;
		loadBinary(in, k);
		loadBinary(in, v);
		m.emplace(k, v);
	}
}

template <typename T>
inline void                loadBinary(std::istream& in, std::set<T>& s)
{
	size_t ssize;
	loadBinary(in, ssize);
	s.clear();
	for (int i = 0; i < ssize; i++)
	{
		T next;
		loadBinary(in, next);
		s.emplace(next);
	}
}

/*template <typename T>
void saveBinary(std::ostream&, T t);

template <>
void                saveBinary(std::ostream&, std::string& s);

template <typename T>
void                saveBinary(std::ostream&, std::vector<T>& v);

template <typename K, typename V>
void                saveBinary(std::ostream&, std::map<K, V>& m);

template <typename T>
void                saveBinary(std::ostream&, std::set<T>& s);

template <typename T>
void loadBinary(std::istream&, T& t);

template <>
void                loadBinary(std::istream&, std::string& s);

template <typename T>
void                loadBinary(std::istream&, std::vector<T>& v);

template <typename K, typename V>
void                loadBinary(std::istream&, std::map<K, V>& m);

template <typename T>
void                loadBinary(std::istream&, std::set<T>& s);*/

}
#endif
