#pragma hdrstop
#include "telException.h"
//---------------------------------------------------------------------------

namespace tlp
{

Exception::Exception(const string& desc)
:
mMessage(desc)//, Message(mMessage)
{
}

Exception::~Exception() noexcept {}

const char* Exception::what() const throw()
{
    return mMessage.c_str();
}

string Exception::Message() const
{
    return mMessage;
}

string Exception::getMessage() const
{
    return mMessage;
}

BadHandleException::BadHandleException(const string& msg)
:
Exception(msg)
{}

BadHandleException::BadHandleException(const string& msg1, const string& msg2)
:
Exception(msg1 + msg2)
{}

BadStringToNumberConversion::BadStringToNumberConversion(std::string const& s)
: Exception(s)
{}

}

