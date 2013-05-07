#ifndef rrExceptionH
#define rrExceptionH
#include <exception>
#include <string>
#include "rrObject.h"

using std::string;
using std::exception;

namespace rr
{

class RR_DECLSPEC Exception : public std::exception, public rrObject
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

class RR_DECLSPEC CoreException : public Exception
{
    public:
        CoreException(const string& msg);
        CoreException(const string& msg1, const string& msg2);
};

class RR_DECLSPEC ScannerException : public Exception
{
    public:
        ScannerException(const string& msg);
};

class RR_DECLSPEC NLEQException : public Exception
{
    public:
        NLEQException(const string& msg);
};

class RR_DECLSPEC NOMException : public Exception
{
    public:
        NOMException(const string& msg);
};

class RR_DECLSPEC CVODEException : public Exception
{
    public:
        CVODEException(const string& msg);
};
}//namepsace rr
#endif
