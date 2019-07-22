#pragma hdrstop
#include "telArrayedParameter.h"
#include <sstream>

namespace rr
{

using namespace std;
ArrayedParameter::ArrayedParameter(const string& lbl, double initialVal, double incr, int nrOfIncrements)
:
mName(lbl),
mInitialValue(initialVal),
mIncrement(incr),
mNumberOfIncrements(nrOfIncrements),
mCurrentValue(mInitialValue)
{}

void ArrayedParameter::increment()
{
    mCurrentValue += mIncrement;
}

void ArrayedParameter::decrement()
{
    mCurrentValue -= mIncrement;
}

string ArrayedParameter::asString() const
{
    stringstream st;
    st<<"Name="<<mName<<"\n";
    st<<"InitialValue="<<mInitialValue<<"\n";
    st<<"Increment="<<mIncrement<<"\n";
    st<<"NumberOfIncrements="<<mNumberOfIncrements<<"\n";
    st<<"CurrentValue="<<mCurrentValue<<"\n";
    return st.str();
}

ostream& operator << (ostream& st, ArrayedParameter& pm)
{
    st<< pm.asString();
    return st;
}

void ArrayedParameter::setNumberOfIncrements(int nr)
{
   mNumberOfIncrements= nr;
}

int ArrayedParameter::getNumberOfIncrements() const
{
    return mNumberOfIncrements;
}

void ArrayedParameter::setIncrement(double incr)
{
    mIncrement = incr;
}

double ArrayedParameter::getIncrement() const
{
    return mIncrement;
}

void ArrayedParameter::setCurrentValue(double val)
{
    mCurrentValue = val;
}

double ArrayedParameter::getCurrentValue() const
{
    return mCurrentValue;
}

void ArrayedParameter::setName(const string& str)
{
    mName = str;
}

string ArrayedParameter::getName() const
{
    return mName;
}

void ArrayedParameter::setInitialValue(double val)
{
    mInitialValue = val;
}

double ArrayedParameter::getInitialValue() const
{
    return mInitialValue;
}

}
