#pragma hdrstop
#include <algorithm>
#include <iostream>
#include <sstream>
#include "telStringUtils.h"
#include "telStringList.h"
#include "telUtils.h"
#include "telException.h"

using namespace std;

namespace tlp
{

using namespace tlp;

StringList::StringList()
{}

StringList::StringList(const vector<string>& strings)
:
mStrings(strings)
{}

//StringList::StringList(RRStringArray* cp)
//{
//    if(!cp)
//    {
//        return;
//    }
//
//    for(int i = 0; i < cp->Count; i++)
//    {
//        mStrings.push_back(cp->String[i]);
//    }
//}

StringList::StringList(const string& str, const string& delimiter)
{
    mStrings = splitString(str, delimiter);
}

StringList::StringList(char** aList, const int& count)
{
    //Copy C stringlist into the container..
    for(int i = 0; i < count; i++)
    {
        mStrings.push_back(aList[i]);
    }
}

StringList::StringList(const StringList& cp)
{
    mStrings = cp.mStrings;
}

StringList::~StringList()
{}

//Converison operator
StringList::operator const vector<string>&() const
{
    return mStrings;
}

void StringList::reSize(int size)
{
    mStrings.resize(size);
}

vector<string>::iterator StringList::begin()
{
    return mStrings.begin();
}

vector<string>::iterator StringList::end()
{
    return mStrings.end();
}

string& StringList::operator[](const size_t& index)
{
    if(index > count() -1 )
    {
        stringstream msg;
        msg<<"index ("<<index<<") out of bounds in StringList with count "<<count();

        throw(Exception(msg.str()));
    }
    return mStrings[index];
}

const string& StringList::operator[](const size_t& index) const
{
    if(index > count() -1 )
    {
        stringstream msg;
        msg<<"index ("<<index<<") out of bounds in StringList with count "<<count();

        throw(Exception(msg.str()));
    }

    return mStrings[index];
}

size_t StringList::size() const
{
    return mStrings.size();
}

size_t StringList::count() const
{
    return mStrings.size();
}

string StringList::asString(const string& delimiter) const
{
    stringstream names;
    for(u_int i = 0; i < mStrings.size(); i++)
    {
        names<<mStrings[i];
        if( i < mStrings.size() - (u_int) 1)
        {
            names<<delimiter;
        }
    }
    return names.str();
}

void StringList::preFix(const string& fix)
{
     for(mLI = mStrings.begin(); mLI != mStrings.end(); mLI++)
    {
        (*mLI) = fix + (*mLI) ;
    }
}

void StringList::postFix(const string& fix)
{
    for(mLI = mStrings.begin(); mLI != mStrings.end(); mLI++)
    {
        (*mLI) = (*mLI) + fix;
    }
}

StringList StringList::operator-(const StringList& rhs)
{
    StringList newList;

    for(u_int i = 0; i < count(); i++)
    {
        string item = mStrings[i] + "-" + rhs[i];
        newList.add(item);
    }

    return newList;
}

void StringList::insertAt(const int& index, const string& item)
{
    mLI = mStrings.begin() + index;
    if(mLI != mStrings.end())
    {
        mStrings.insert(mLI, item);
    }
}

void StringList::append(const StringList& list)
{
    for(u_int i = 0; i < list.count(); i++)
    {
        mStrings.push_back(list[i]);
    }
}

void StringList::add(const string& item)
{
    mStrings.push_back(item);
}

ptrdiff_t StringList::find(const string& item)
{
    return tlp::indexOf(mStrings, item);
}

ptrdiff_t StringList::indexOf(const string& item)
{
    return tlp::indexOf(mStrings, item);
}

void StringList::removeAt(const int& index)
{
    mLI = mStrings.begin() + index;
    mStrings.erase(mLI);
}

bool StringList::contains(const string& item) const
{
    return std::find(mStrings.begin(), mStrings.end(), item) != mStrings.end() ? true : false;
}

bool StringList::dontContain(const string& item) const
{
    return !contains(item);
}

bool isRHSLonger(string const& lhs, string const& rhs)
{
    return lhs.size() < rhs.size();
}

string StringList::getLongest()
{
	mLI = std::max_element(mStrings.begin(), mStrings.end(), isRHSLonger);
    return *mLI;
}

string StringList::getShortest()
{
    mLI = std::min_element(mStrings.begin(), mStrings.end(), isRHSLonger);
    return *mLI;
}

void StringList::clear()
{
    mStrings.clear();
}

void StringList::empty()
{
    mStrings.clear();
}

StringList& StringList::operator=(const StringList& rhs)
{
    mStrings = rhs.mStrings;
    return *this;
}

StringList& StringList::operator=(const vector<string>& rhs)
{
    mStrings = rhs;
    return *this;
}

ostream& operator<<(ostream& stream, const StringList& list)
{
    stream<<"{";
    for(u_int i = 0; i < list.count(); i++)
    {
        stream<<"\""<<list[i]<<"\"";
        if(i < list.count() - (u_int) 1)
        {
            stream<<",";
        }

    }
    stream<<"}";
    return stream;
}


} //namespace tlp

