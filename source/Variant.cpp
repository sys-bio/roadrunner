/*
 * Variant.cpp
 *
 *  Created on: Apr 26, 2014
 *      Author: andy
 */

#include "Variant.h"
#include "rrLogger.h"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <cctype>
#include <cstdlib>
#include <algorithm>
#include <assert.h>
#include <Poco/Dynamic/Var.h>
#include <stdint.h>
#include <source\rrStringUtils.h>

using namespace std;
using Poco::Dynamic::Var;

namespace rr
{

struct VariantImpl
{
    Var var;
    unsigned size;
};

Variant::Variant()
{
    alloc();
}

Variant::Variant(const Variant& other)
{
    alloc();
    self->var = other.self->var;
    self->size = other.self->size;
}

Variant& Variant::operator =(const Variant& other)
{
    self->var = other.self->var;
    self->size = other.self->size;
    return *this;
}

Variant::~Variant()
{
    delete self;
}

const std::type_info& Variant::typeInfo() const
{
    return self->var.type();
}

void Variant::alloc()
{
    self = new VariantImpl();
}

#define TRY_ASSIGN(type)                                    \
        if (info == typeid(type)) {                         \
            const type *val = static_cast<const type*>(p);  \
            self->var = *val;                               \
            self->size = sizeof(type);                      \
            return;                                         \
        }

void Variant::assign(const std::type_info& info, const void* p)
{
    TRY_ASSIGN(std::string);

    TRY_ASSIGN(long);

    TRY_ASSIGN(bool);

    TRY_ASSIGN(float);

    TRY_ASSIGN(double);

    TRY_ASSIGN(unsigned long);

    TRY_ASSIGN(int);

    TRY_ASSIGN(unsigned int);

    TRY_ASSIGN(char);

    TRY_ASSIGN(unsigned char);

    TRY_ASSIGN(int32_t);

    TRY_ASSIGN(uint32_t);

    TRY_ASSIGN(int64_t);

    TRY_ASSIGN(uint64_t);

	TRY_ASSIGN(vector<double>);

	TRY_ASSIGN(vector<string>);

    string msg = "could not assign type ";
    msg += info.name();
    msg += " to Variant";

    throw invalid_argument(msg);
}

/**
 * strip any leading or trailing whitespace
 */
static std::string strip(const std::string& in)
{
    std::string out;
    std::string::const_iterator b = in.begin(), e = in.end();

    // skipping leading spaces
    while (std::isspace(*b)){
        ++b;
    }

    if (b != e){
        // skipping trailing spaces
        while (std::isspace(*(e-1))){
            --e;
        }
    }

    out.assign(b, e);

    return out;
}


/*
* Converts a string of the form "[double, double, ..., double]" to a vector
* of doubles.
* Pre: The string must start with '[' (with no preceding spaces) and end with ']',
*      The doubles in the string must be separated by commas and be able to
*      be parsed with std::stod
*
*/
std::vector<double> parseDoubleVector(const std::string& toParse) {
	std::vector<double> result;
	size_t index = 1;
	while (toParse[index] == ' ') {
		index++;
	}
	while (index < toParse.length() && toParse[index] != ']') {
		size_t lengthInStr = 0;
		result.push_back(std::stod(&toParse[index], &lengthInStr));
		index += lengthInStr;
		while (index < toParse.length() && (toParse[index] == ' '
			|| toParse[index] == ',')) {
			index++;
		}
	}
	return result;
}

Variant Variant::parse(const std::string& s)
{
    string str = strip(s);

    const char* input = str.c_str();
    char* end = 0;

    // check for int
    int i = strtol(input, &end, 0);
    if (*input != '\0' && end != input && *end == '\0')
    {
        return Variant(i);
    }

    // check for double
    double d = strtod(input, &end);
    if (*input != '\0' && end != input && *end == '\0')
    {
        return Variant(d);
    }

    // check for bool
    std::string bstr = str;
    std::transform(bstr.begin(), bstr.end(),bstr.begin(), ::toupper);

    if (bstr == "TRUE") {
        return Variant(true);
    }

    if (bstr == "FALSE") {
        return Variant(false);
    }
	
	// check if it is a vector
	if (str[0] == '[') {
		string vstr = str.substr(1, str.size() - 1);
		vector<string> parts(splitString(vstr, ","));

		// check if it is a double vector
		const char* input = parts[0].c_str();
		char* end = 0;
		double d = strtod(input, &end);
		if (*input != '\0' && end != input && *end == '\0')
		{
			return Variant(parseDoubleVector(str));
		}

		// no, it is a string vector
		return Variant(parts);
	}

    // its a string
    Variant result;
    result.self->var = Var::parse(str);
    return result;
}

std::string Variant::toString() const
{
    return Var::toString(self->var);
}

std::string Variant::pythonRepr() const
{
    if (isBool()) {
        switch (convert<bool>()) {
          case true:
              return "True";
          case false:
              return "False";
        };
    } else if (isString()) {
        return "'" + toString() + "'";
    } else
        return toString();
}

bool Variant::isString() const
{
    return self->var.isString();
}

bool Variant::isInteger() const
{
    return self->var.isInteger();
}

bool Variant::isNumeric() const
{
    return self->var.isNumeric();
}

bool Variant::isBool() const
{
    return self->var.type() == typeid(bool);
}

bool Variant::isDoubleVector() const 
{
	return self->var.type() == typeid(vector<double>);
}

bool Variant::isStringVector() const
{
	return self->var.type() == typeid(vector<string>);
}



#define TRY_CONVERT_TO(type)                       \
        if (info == typeid(type)) {                \
            type* out = static_cast<type*>(p);     \
            *out = self->var.convert<type>();      \
            return;                                \
        }

bool Variant::isEmpty() const
{
    return self->var.isEmpty();
}

bool Variant::isSigned() const
{
    return self->var.isSigned();
}

#define TYPE_KIND(t, tid) if (info == typeid(t)) return tid;

Variant::TypeId Variant::type() const
{
    const std::type_info& info = self->var.type();
    TYPE_KIND(std::string, STRING);
    TYPE_KIND(int32_t, INT32);
    TYPE_KIND(uint32_t, UINT32);
    TYPE_KIND(int64_t, INT64);
    TYPE_KIND(uint64_t, UINT64);
    TYPE_KIND(float, FLOAT);
    TYPE_KIND(double, DOUBLE);
    TYPE_KIND(char, CHAR);
    TYPE_KIND(unsigned char, UCHAR);
    TYPE_KIND(bool, BOOL);
	TYPE_KIND(vector<double>, DOUBLEVECTOR);
	TYPE_KIND(vector<string>, STRINGVECTOR);


    if(info == typeid(int)) {
        if(self->size == 4) return INT32;
        if(self->size == 8) return INT64;
    }

    if(info == typeid(long)) {
        if(self->size == 4) return INT32;
        if(self->size == 8) return INT64;
    }

    if(info == typeid(unsigned int)) {
        if(self->size == 4) return UINT32;
        if(self->size == 8) return UINT64;
    }

    if(info == typeid(unsigned long)) {
        if(self->size == 4) return UINT32;
        if(self->size == 8) return UINT64;
    }


    return EMPTY;
}

void Variant::convert_to(const std::type_info& info, void* p) const
{
    try
    {
        TRY_CONVERT_TO(std::string);

        TRY_CONVERT_TO(long);

        TRY_CONVERT_TO(bool);

        TRY_CONVERT_TO(float);

        TRY_CONVERT_TO(double);

        TRY_CONVERT_TO(unsigned long);

        TRY_CONVERT_TO(int);

        TRY_CONVERT_TO(unsigned int);

        TRY_CONVERT_TO(char);

        TRY_CONVERT_TO(unsigned char);

        TRY_CONVERT_TO(int32_t);

        TRY_CONVERT_TO(uint32_t);

		TRY_CONVERT_TO(std::vector<double>);
		TRY_CONVERT_TO(std::vector<string>);

		
		//if (info == typeid(std::vector<double>)) {
		//	std::vector<double>* out = static_cast<std::vector<double>*>(p);
		//	*out = self->var.extract<std::vector<double>>();
		//	return;
		//}

		//if (info == typeid(std::vector<string>)) {
		//	std::vector<string>* out = static_cast<std::vector<string>*>(p);
		//	*out = self->var.extract<std::vector<string>>();
		//	return;
		//}

    }
    catch(Poco::SyntaxException& ex)
    {
        string msg = "Could not convert variant with typeid ";
        msg += self->var.type().name();
        msg += " to type";
        msg += info.name();
        msg += "error: ";
        msg += ex.what();
        msg += ", string value: " + self->var.toString();

        throw std::logic_error(msg);
    }


    string msg = "Could not convert variant with typeid ";
    msg += self->var.type().name();
    msg += " to type";
    msg += info.name();

    throw invalid_argument(msg);
}

} /* namespace rr */


