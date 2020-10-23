#ifndef telArrayListH
#define telArrayListH
#include <vector>
#include <string>
#include <list>
#include <ostream>
#include "telCommonExporter.h"
namespace tlp
{

class StringList;
using tlp::StringList;
using std::vector;
using std::string;

class COMMON_DECLSPEC ArrayListItemBase
{
    public:
        virtual ~ArrayListItemBase();
};

class COMMON_DECLSPEC ArrayList
{
    protected:
    public:
        vector< ArrayListItemBase* >   mList; //List of ArrayListItemBase items

    public:
                                            ArrayList();
                                            ArrayList(const ArrayList& cpyMe);
                                            ArrayList(const string& lbl, const StringList& stringList);
                                            ArrayList(const string& lbl, const ArrayList& stringList);
                                           ~ArrayList();
        size_t                              Count() const;
        void                                Clear();
        void                                Add(const int& item);
        void                                Add(const double& item);
        void                                Add(const string& item);
        void                                Add(const ArrayList& item);
        void                                Add(const StringList& list);
        void                                Add(const string& lbl, const StringList& list);
        void                                Add(const string& lbl, const ArrayList& list);


        const ArrayListItemBase&       operator[](int pos) const;
        ArrayListItemBase&             operator[](int pos);
        void                                operator = (const ArrayList& rhs);
        StringList                          GetStringList(const string& lName);
        StringList                          GetStringList(const int& index);
        string                              GetString(const int& index);
        string                              AsString();
};


COMMON_DECLSPEC std::ostream& operator<<(std::ostream& stream, const ArrayList& list);



template <class T>
class ArrayListItem : public ArrayListItemBase
{
    private:
        T                           mItemValue;

    public:
                                    ArrayListItem(const T& val);

        virtual                    ~ArrayListItem(){}
                                    operator T(){return mItemValue;}
        virtual const char          operator[](const int& pos) const {return '\0';}     //Make sense for string types
        ArrayListItem<T>&           operator=(const ArrayListItem<T>& rhs);
};

template<class T>
ArrayListItem<T>::ArrayListItem(const T& val)
:
mItemValue(val)
{}

template<class T>
ArrayListItem<T>& ArrayListItem<T>::operator=(const ArrayListItem<T>& rhs)
{
    if(this != &rhs)
    {
        mItemValue = rhs.mItemValue;
    }

    return *this;
}

std::ostream& operator<<(std::ostream& stream, const ArrayListItemBase& item);

}
#endif
