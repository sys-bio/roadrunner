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
#include "Setting.h"

namespace rr {
    using std::string;
    using std::list;
    using std::vector;
    using std::complex;

    RR_DECLSPEC char *createText(const std::string &str);

    RR_DECLSPEC char *createText(const int &count);

    RR_DECLSPEC bool freeText(char *str);

    RR_DECLSPEC std::string replaceWord(const std::string &str1, const std::string &str2, const std::string &theString);

    RR_DECLSPEC bool convertFunctionCallToUseVarArgsSyntax(const std::string &funcName, std::string &expression);

    RR_DECLSPEC std::string removeChars(const std::string &str, const std::string &chars);

    RR_DECLSPEC bool isUnwantedChar(char ch); //Predicate for find_if algorithms..
    RR_DECLSPEC size_t findMatchingRightParenthesis(const std::string &expression, const size_t startFrom);

    RR_DECLSPEC int getNumberOfFunctionArguments(const std::string &expression);

    RR_DECLSPEC std::string tabs(const int &nr);

    RR_DECLSPEC std::string NL();

    RR_DECLSPEC std::string toUpperOrLowerCase(const std::string &inStr, int (*func)(int));

    RR_DECLSPEC std::string toUpper(const std::string &str);

    RR_DECLSPEC std::string toLower(const std::string &str);

    RR_DECLSPEC std::string getFilePath(const std::string &fileN);

    RR_DECLSPEC std::string getFileName(const std::string &fileN);

    RR_DECLSPEC std::string getFileNameNoExtension(const std::string &fileN);

    RR_DECLSPEC std::string getFileExtension(const std::string &fileN);

    RR_DECLSPEC std::string changeFileExtensionTo(const std::string &theFileName, const std::string &newExtension);

    RR_DECLSPEC int compareNoCase(const std::string &str1, const std::string &str2);

    RR_DECLSPEC std::string trim(const std::string &str, const char &toTrim);

    /**
     * @brief removes both leading and trailing white
     * space characters from input string @param str
     */
    RR_DECLSPEC std::string& trim(std::string &str);

    /**
     * @brief left trim. Removes leading white space characters
     * from string @param str
     */
    RR_DECLSPEC std::string & ltrim(std::string & str);

    /**
     * @brief right trim. Removes trailing white space characters
     * from string @param str
     */
    RR_DECLSPEC std::string & rtrim(std::string & str);

    RR_DECLSPEC bool startsWith(const std::string &src, const std::string &sub);

    RR_DECLSPEC bool endsWith(const std::string &src, const std::string &sub);

//conversions
    RR_DECLSPEC std::string intToStr(const int &nt);

    RR_DECLSPEC int strToInt(const std::string &nt);

    RR_DECLSPEC std::string dblToStr(const double &nt);

    RR_DECLSPEC double strToDbl(const std::string &nt);

    RR_DECLSPEC std::vector<std::string> splitString(const std::string &text, const std::string &separators);

    RR_DECLSPEC std::vector<std::string> splitString(const std::string &input, const char &delimiters);

    RR_DECLSPEC int toInt(const std::string &str);

    RR_DECLSPEC bool toBool(const std::string &str);

    RR_DECLSPEC double toDouble(const std::string &str);

    RR_DECLSPEC std::vector<double> toDoubleVector(const std::string &str);

    RR_DECLSPEC complex<double> toComplex(const std::string &str);

    RR_DECLSPEC std::string toString(const bool &b);

    RR_DECLSPEC std::string toString(const double &d, const std::string &format = gDoubleFormat);

    RR_DECLSPEC std::string
    toString(const unsigned int &n, const std::string &format = gIntFormat, const int nBase = 10);

    RR_DECLSPEC std::string toString(const int &n, const std::string &format = gIntFormat, const int nBase = 10);

    RR_DECLSPEC std::string toString(const long n, const int nBase = 10);

    RR_DECLSPEC std::string toString(const unsigned long n, const int nBase = 10);

    RR_DECLSPEC std::string toString(const unsigned short n, const int nBase = 10);

    RR_DECLSPEC std::string toString(const short n, const int nBase = 10);

    RR_DECLSPEC std::string toString(const char n);

    RR_DECLSPEC std::string toString(const unsigned char n);

    RR_DECLSPEC std::string toString(const std::string &s);

    RR_DECLSPEC std::string toString(const char *str);

    RR_DECLSPEC std::string toString(const std::vector<std::string> &vec, const std::string &sep = ", ");

    RR_DECLSPEC std::string toStringSize(size_t n);

    RR_DECLSPEC std::string format(const std::string &src, const int &arg);

    RR_DECLSPEC std::string format(const std::string &str, const int &arg1);


    RR_DECLSPEC std::string format(const std::string &src, const std::string &arg);

    RR_DECLSPEC std::string
    format(const std::string &src, const std::string &arg1, const std::string &arg2, const std::string &arg3);

    RR_DECLSPEC std::string format(const std::string &src, const std::string &arg1, const std::string &arg2);

    RR_DECLSPEC std::string format(const std::string &src, const std::string &arg1, const int &arg2);

    RR_DECLSPEC std::string
    format(const std::string &src, const std::string &arg1, const int &arg2, const std::string &arg3);

    RR_DECLSPEC std::string format(const std::string &str1, const std::string &str2);

    RR_DECLSPEC std::string format(const std::string &str1, const std::string &arg1, const std::string &arg2);

    RR_DECLSPEC std::string format(const std::string &str1, const std::string &arg1, const int &arg2);

    RR_DECLSPEC std::string
    format(const std::string &str1, const std::string &arg1, const int &arg2, const std::string &arg3);

    RR_DECLSPEC std::string
    format(const std::string &str1, const std::string &arg1, const std::string &arg2, const std::string &arg3);

    RR_DECLSPEC std::string
    format(const std::string &str1, const std::string &arg1, const std::string &arg2, const std::string &arg3,
           const std::string &arg4);

    RR_DECLSPEC std::string
    format(const std::string &str1, const std::string &arg1, const std::string &arg2, const std::string &arg3,
           const std::string &arg4, const std::string &arg5);

    RR_DECLSPEC std::string format(const std::string &str1, const unsigned int &arg1, const std::string &arg2);

    RR_DECLSPEC std::string
    format(const std::string &str1, const unsigned int &arg1, const std::string &arg2, const std::string &arg3);

    RR_DECLSPEC std::string
    format(const std::string &str1, const unsigned int &arg1, const unsigned int &arg2, const std::string &arg3,
           const std::string &arg4);

    RR_DECLSPEC std::string append(const std::string &str);

    RR_DECLSPEC std::string append(const int &str);

    RR_DECLSPEC std::string append(const unsigned int &str);

    RR_DECLSPEC std::string append(const std::string &s1, const std::string &s2);

    RR_DECLSPEC std::string append(const std::string &s1, const std::string &s2, const std::string &s3);

    RR_DECLSPEC std::string append(const std::string &s1, const unsigned int &s2, const std::string &s3);

    RR_DECLSPEC std::string
    append(const std::string &s1, const unsigned int &s2, const std::string &s3, const std::string &s4);

    RR_DECLSPEC std::string
    substitute(const std::string &src, const std::string &thisOne, const std::string &withThisOne,
               const int &howMany = -1);

    RR_DECLSPEC std::string
    substitute(const std::string &src, const std::string &thisOne, const int &withThisOne, const int &howMany = -1);
    //RR_DECLSPEC std::string              substitute(const std::string& src, const std::string& thisOne, const double& withThisOne, const int& howMany = -1);
    RR_DECLSPEC std::string removeNewLines(const std::string &str, const int &howMany = -1);


    /*
    * Saves t in a binary format that can then be reloaded with loadBinary
    * T must be a primitive, POD with no pointers, std::vector, std::string, std::map, std::unordered_map, std::set, rr::Setting
    * or combinations of those, like std::map<rr::Setting, std::vector<std::string> >
    */
    template<typename T>
    inline void saveBinary(std::ostream &out, const T &t) {
        out.write((char *) &t, sizeof(T));
    }

    template<>
    inline void saveBinary(std::ostream &out, const std::string &s) {
        saveBinary(out, s.size());
        out.write(s.c_str(), s.size());
    }

    template<typename T>
    inline void saveBinary(std::ostream &out, const std::vector<T> &v) {
        saveBinary(out, v.size());
        for (T t : v) {
            saveBinary(out, t);
        }
    }

    template<typename K, typename V>
    inline void saveBinary(std::ostream &out, const std::map<K, V> &m) {
        saveBinary(out, m.size());
        for (std::pair<K, V> p : m) {
            saveBinary(out, p.first);
            saveBinary(out, p.second);
        }
    }

    template<typename K, typename V>
    inline void saveBinary(std::ostream &out, const std::unordered_map<K, V> &m) {
        saveBinary(out, m.size());
        for (std::pair<K, V> p : m) {
            saveBinary(out, p.first);
            saveBinary(out, p.second);
        }
    }

    template<typename T>
    inline void saveBinary(std::ostream &out, const std::set<T> &s) {
        saveBinary(out, s.size());
        for (T t : s) {
            saveBinary(out, t);
        }
    }

    template<>
    inline void saveBinary<rr::Setting>(std::ostream &out, const rr::Setting &var) {
        saveBinary(out, var.type());
        switch (var.type()) {
            case Setting::BOOL:
                saveBinary(out, var.get<bool>());
                break;
            case Setting::CHAR:
                saveBinary(out, var.get<char>());
                break;
            case Setting::DOUBLE:
                saveBinary(out, var.get<double>());
                break;
            case Setting::FLOAT:
                saveBinary(out, var.get<float>());
                break;
            case Setting::INT32:
                saveBinary(out, var.get<std::int32_t>());
                break;
            case Setting::INT64:
                saveBinary(out, var.get<std::int64_t>());
                break;
            case Setting::STRING:
                saveBinary(out, var.get<std::string>());
                break;
            case Setting::UCHAR:
                saveBinary(out, var.get<unsigned char>());
                break;
            case Setting::UINT32:
                saveBinary(out, var.get<std::uint32_t>());
                break;
            case Setting::UINT64:
                saveBinary(out, var.get<std::uint64_t>());
                break;
            case Setting::DOUBLEVECTOR:
                saveBinary(out, var.get<std::vector<double>>());
            default:
                break;
        }
    }

    /*
    * Loads an object saved by saveBinary from an std::istream and puts the result in t
    * t must be of the same type that was saved by saveBinary
    * Objects must be loaded in the order they were saved to the stream
    */
    template<typename T>
    inline void loadBinary(std::istream &in, T &t) {
        T temp;
        in.read((char *) &temp, sizeof(T));
        t = temp;
    }

    template<>
    inline void loadBinary(std::istream &in, std::string &s) {
        size_t ssize;
        loadBinary(in, ssize);
        s.resize(ssize);
        in.read(&s[0], ssize);
    }

    template<typename T>
    inline void loadBinary(std::istream &in, std::vector<T> &v) {
        size_t vsize;
        loadBinary(in, vsize);
        v.clear();
        for (int i = 0; i < vsize; i++) {
            T next;
            loadBinary(in, next);
            v.push_back(next);
        }
    }

    template<typename K, typename V>
    inline void loadBinary(std::istream &in, std::map<K, V> &m) {
        size_t msize;
        loadBinary(in, msize);
        m.clear();
        for (int i = 0; i < msize; i++) {
            K k;
            V v;
            loadBinary(in, k);
            loadBinary(in, v);
            m.emplace(k, v);
        }
    }

    template<typename K, typename V>
    inline void loadBinary(std::istream &in, std::unordered_map<K, V> &m) {
        size_t msize;
        loadBinary(in, msize);
        m.clear();
        for (int i = 0; i < msize; i++) {
            K k;
            V v;
            loadBinary(in, k);
            loadBinary(in, v);
            m.emplace(k, v);
        }
    }

    template<typename T>
    inline void loadBinary(std::istream &in, std::set<T> &s) {
        size_t ssize;
        loadBinary(in, ssize);
        s.clear();
        for (int i = 0; i < ssize; i++) {
            T next;
            loadBinary(in, next);
            s.emplace(next);
        }
    }

    template<>
    inline void loadBinary<rr::Setting>(std::istream &in, rr::Setting &var) {
    	Setting::TypeId type;
        std::string strVal;
    	std::vector<double> vectorVal;
    	loadBinary(in, type); // what is this for?

    	// todo use visitor here
	switch (type)
	{
	case Setting::BOOL:
		bool boolVal;
		loadBinary(in, boolVal);
		var = Setting(boolVal);
		break;
	case Setting::CHAR:
        char charVal;
		loadBinary(in, charVal);
		var = Setting(charVal);
		break;
	case Setting::DOUBLE:
        double doubleVal;
		loadBinary(in, doubleVal);
		var = Setting(doubleVal);
		break;
	case Setting::FLOAT:
        float floatVal;
		loadBinary(in, floatVal);
		var = Setting(floatVal);
		break;
	case Setting::INT32:
        std::int32_t int32Val;
		loadBinary(in, int32Val);
		var = Setting(int32Val);
		break;
	case Setting::INT64:
        std::int64_t int64Val;
		loadBinary(in, int64Val);
		var = Setting(int64Val);
		break;
	case Setting::STRING:
		loadBinary(in, strVal);
		var = Setting(strVal);
		break;
	case Setting::UCHAR:
        unsigned char ucharVal;
		loadBinary(in, ucharVal);
		var = Setting(ucharVal);
		break;
	case Setting::UINT32:
        std::uint32_t uint32Val;
		loadBinary(in, uint32Val);
		var = Setting(uint32Val);
		break;
	case Setting::UINT64:
        std::uint64_t uint64Val;
		loadBinary(in, uint64Val);
		var = Setting(uint64Val);
		break;
	case Setting::DOUBLEVECTOR:
		loadBinary(in, vectorVal);
		var = Setting(vectorVal);
		break;
	default:
		break;
	}

	// todo replace the above logic with something like the following.
	// remember that get_if returns a pointer!!!
//        if (auto monostateVal = var.get_if<std::monostate>()) {
//            // std::monostate is an alias for void in a std::variant (which cannot have void types).
//            // Therefore we just do nothing if var is a std::monostate
//        } else if (auto strVal = var.get_if<std::string>()) {
//            loadBinary(in, *strVal);
//        } else if (auto boolVal = var.get_if<bool>()) {
//            loadBinary(in, *boolVal);
//        } else if (auto int32Val = var.get_if<std::int32_t>()) {
//            loadBinary(in, *int32Val);
//        } else if (auto uInt32Val = var.get_if<std::uint32_t>()) {
//            loadBinary(in, *uInt32Val);
//        } else if (auto int64Val = var.get_if<std::int64_t>()) {
//            loadBinary(in, *int64Val);
//        } else if (auto uInt64Val = var.get_if<std::uint64_t>()) {
//            loadBinary(in, *uInt64Val);
//        } else if (auto floatVal = var.get_if<float>()) {
//            loadBinary(in, *floatVal);
//        } else if (auto doubleVal = var.get_if<double>()) {
//            loadBinary(in, *doubleVal);
//        } else if (auto charVal = var.get_if<char>()) {
//            loadBinary(in, *charVal);
//        } else if (auto uChar = var.get_if<unsigned char>()) {
//            loadBinary(in, *uChar);
//        } else if (auto doubleVector = var.get_if<std::vector<double>>()) {
//            loadBinary(in, *doubleVector);
//        } else {
//            throw std::invalid_argument("Setting is not a valid type");
//        }
    }

}
#endif
