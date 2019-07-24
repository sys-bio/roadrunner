#pragma hdrstop
#include "rrException.h"
//---------------------------------------------------------------------------


namespace rr
{

Exception::Exception(const string& desc)
:
mMessage(desc)//, Message(mMessage)
{
}

Exception::~Exception() throw() {}

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

CoreException::CoreException(const string& msg)
:
Exception(msg)
{}

CoreException::CoreException(const string& msg1, const string& msg2)
:
Exception(msg1 + msg2)
{}

ScannerException::ScannerException(const string& msg)
:
Exception(msg)
{}

NLEQException::NLEQException(const string& msg)
:
Exception(msg)
{}

NOMException::NOMException(const string& msg)
:
Exception(msg)
{}

CVODEException::CVODEException(const string& msg)
:
Exception(msg)
{}

NotImplementedException::NotImplementedException(const string& msg)
:
Exception(msg)
{}

InvalidKeyException::InvalidKeyException(const string& msg)
:
Exception(msg)
{}

UninitializedValueException::UninitializedValueException(const string& msg)
:
Exception(msg)
{}

void UninitializedValue(const string& msg) {
    throw UninitializedValueException(msg);
}

BadStringToNumberConversion::BadStringToNumberConversion(std::string const& s)
: Exception(s)
{}

}
