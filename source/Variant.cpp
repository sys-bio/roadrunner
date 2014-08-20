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

using namespace std;
using Poco::Dynamic::Var;

namespace rr
{

struct VariantImpl
{
    Var var;
};

Variant::Variant()
{
    alloc();
}

Variant::Variant(const Variant& other)
{
    alloc();
    self->var = other.self->var;
}

Variant& Variant::operator =(const Variant& other)
{
    self->var = other.self->var;
    return *this;
}

Variant::~Variant()
{
    delete self;
}

const std::type_info& Variant::type() const
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

    // its a string
    Variant result;
    result.self->var = Var::parse(str);
    return result;
}

std::string Variant::toString() const
{
    return Var::toString(self->var);
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


