#pragma hdrstop
#include <sstream>
#include "telStringListContainer.h"

using namespace std;

namespace tlp
{

StringListContainer::StringListContainer()
{
}

StringListContainer::StringListContainer(const StringListContainer& cp)
:
mContainer(cp.mContainer)
{}

//StringListContainer::StringListContainer(const string& lbl, const StringListContainer& cp)
//:
//mLabel(lbl),
//mContainer(cp.mContainer)
//{}

StringListContainer::StringListContainer(const StringList& cp)
{
    Add(cp);
}

size_t StringListContainer::TotalCount() const
{
    //Returns the total count of all list items..
    size_t cnt = 0;
    for(int i = 0; i < Count(); i++)
    {
        cnt += mContainer[i].count();
    }
    return cnt;
}

size_t StringListContainer::ListCount() const
{
    return mContainer.size();
}

size_t StringListContainer::Count() const
{
    return mContainer.size();
}

StringList& StringListContainer::operator[](const int& index)
{
    return mContainer[index];
}

const StringList& StringListContainer::operator[](const int& index) const
{
    return mContainer[index];
}

vector<StringList>::iterator StringListContainer::begin()
{
    return mContainer.begin();
}

vector<StringList>::iterator StringListContainer::end()
{
    return mContainer.end();
}

//void StringListContainer::Add(const string& lbl, const StringListContainer& lists)
//{
//    mLabel = lbl;
//    Add(lists);
//}

void StringListContainer::Add(const StringListContainer& lists)
{
    for(int i = 0; i < lists.Count(); i++)
    {
        StringList aList;
        aList = lists.mContainer[i];    //Todo: lists[i] should work...
        Add(aList);
    }
}

void StringListContainer::Add(const StringList& list)
{
    mContainer.push_back(list);
}

void StringListContainer::Add(const string& listName, const StringList& aList)
{
    StringList list(aList);
//    list.Label(listName);
    mContainer.push_back(list);
}

void StringListContainer::Add(const string& item)
{
    StringList list;
    list.add(item);
    Add(list);

}

void StringListContainer::Add(const int& atPos)
{

}

ostream& operator<<(ostream& stream, const StringListContainer& list)
{
    vector<StringList>::iterator iter;
    for(int  i = 0; i < list.Count(); i++)
    {
        string item = list[i].asString();
        stream<<"List Item "<<i+1<<" : "<<item<<endl;
    }
    return stream;
}

}

