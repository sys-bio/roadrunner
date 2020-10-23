#ifndef telExceptionH
#define telExceptionH
#include <exception>
#include <string>
#include "telCommonExporter.h"

using std::string;
using std::exception;

namespace tlp
{

class Exception : public std::exception
{
    protected:
        string mMessage;   //Exception message

    public:
        //string& Message;
                                Exception(const string& desc);
        virtual                ~Exception() throw();
        virtual const char*     what() const throw();
        string                  Message() const;
        string                  getMessage() const;
};

class BadHandleException : public Exception
{
    public:
        BadHandleException(const string& msg);
        BadHandleException(const string& msg1, const string& msg2);
};

class BadStringToNumberConversion : public Exception
{
    public:
        BadStringToNumberConversion(std::string const& s);
};



}//namepsace rr
#endif
