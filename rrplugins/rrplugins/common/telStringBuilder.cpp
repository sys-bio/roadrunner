#pragma hdrstop
#include <iomanip>
#include "telStringBuilder.h"
#include "telStringUtils.h"
#include "telLogger.h"
//---------------------------------------------------------------------------

using namespace std;
namespace tlp
{

StringBuilder::StringBuilder(const string& aStr)
{
    if(aStr != gEmptyString)
    {
        mStringing<<aStr;
    }
}

string StringBuilder::ToString()
{
    return mStringing.str();
}

void StringBuilder::Clear()
{
    mStringing.str("");
}

stringstream& StringBuilder::operator<<(const string& str)
{
    mStringing<<str;
    RRPLOG(lDebug5)<<"Appended :"<<removeNewLines(str, 1);
    return mStringing;
}

stringstream& StringBuilder::operator<<(const char& ch)
{
    mStringing<<ch;
    RRPLOG(lDebug5)<<"Appended :"<<ch;
    return mStringing;
}

stringstream& StringBuilder::operator<<(const int& ch)
{
    mStringing<<ch;
    RRPLOG(lDebug5)<<"Appended :"<<ch;
    return mStringing;
}

void StringBuilder::NewLine(const string& line)
{
    mStringing<<"\n"<<line<<endl;
}

void StringBuilder::AppendLine(const string& line)
{
    mStringing<<"\n"<<line<<endl;
}


void StringBuilder::Line(const string& line)
{
    mStringing<<line<<endl;
}

void StringBuilder::TLine(const string& line, const int& nrTabs)
{
    string tabs;
    for(int i = 0; i < nrTabs; i++)
    {
        tabs +="\t";
    }

    mStringing<<tabs<<line<<endl;
}

}
