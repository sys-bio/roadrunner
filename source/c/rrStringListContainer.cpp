#pragma hdrstop
#include <sstream>
#include "rrStringListContainer.h"



namespace rr
{

StringListContainer::StringListContainer()
{
}

StringListContainer::StringListContainer(const StringListContainer& cp)
:
mContainer(cp.mContainer)
{}

//StringListContainer::StringListContainer(const std::string& lbl, const StringListContainer& cp)
//:
//mLabel(lbl),
//mContainer(cp.mContainer)
//{}

StringListContainer::StringListContainer(const StringList& cp)
{
    Add(cp);
}

int StringListContainer::TotalCount() const
{
    //Returns the total count of all list items..
    int cnt = 0;
    for(int i = 0; i < Count(); i++)
    {
        cnt += mContainer[i].Count();
    }
    return cnt;
}

int StringListContainer::ListCount() const
{
    return mContainer.size();
}

int StringListContainer::Count() const
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

std::vector<StringList>::iterator StringListContainer::begin()
{
    return mContainer.begin();
}

std::vector<StringList>::iterator StringListContainer::end()
{
    return mContainer.end();
}

//void StringListContainer::Add(const std::string& lbl, const StringListContainer& lists)
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

void StringListContainer::Add(const std::string& listName, const StringList& aList)
{
    StringList list(aList);
//    list.Label(listName);
    mContainer.push_back(list);
}

void StringListContainer::Add(const std::string& item)
{
    StringList list;
    list.add(item);
    Add(list);

}

void StringListContainer::Add(const int& atPos)
{

}

std::ostream& operator<<(std::ostream& stream, const StringListContainer& list)
{
    std::vector<StringList>::iterator iter;
    for(int  i = 0; i < list.Count(); i++)
    {
        std::string item = list[i].AsString();
        stream<<"List Item "<<i+1<<" : "<<item<<std::endl;
    }
    return stream;
}

}

