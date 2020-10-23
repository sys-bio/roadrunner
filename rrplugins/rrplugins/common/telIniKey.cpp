#pragma hdrstop
#include "telIniKey.h"

using namespace std;
namespace tlp
{

IniKey::IniKey(const string& key)
{
    SetupKey(key);
}

void IniKey::ReKey(const string& key)
{
    SetupKey(key);
}

ostream&  operator<<(ostream& stream, const IniKey& aKey)
{
    stream<<aKey.AsString();
    return stream;
}

string IniKey::AsString() const
{
    string tmp = mKey;
    tmp += " = ";
    tmp += mValue;
    return tmp;
}

int IniKey::AsBool() const
{
    return toBool(mValue);
}

int IniKey::AsInt() const
{
    return toInt(mValue);
}

double IniKey::AsFloat() const
{
    return toDouble(mValue);
}

complex<double> IniKey::AsComplex() const
{
    vector<string> vals = splitString(mValue,",");
    if(vals.size() == 2)
    {
        return complex<double>(toDouble(vals[0]), toDouble(vals[1]));
    }
    return complex<double>(toDouble(mValue), 0 );
}

void IniKey::SetupKey(const string& key)
{
    if(key.size())
    {
        vector<string> recs = splitString(key, "=");
        if(recs.size() > 0)
        {
            mKey = recs[0];
            mValue = recs[1];
          }
        mComment = "";
    }
    else
    {
        mKey = "";
        mValue = "";
        mComment = "";
    }
}

}
