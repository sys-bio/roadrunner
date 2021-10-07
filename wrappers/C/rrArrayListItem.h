#ifndef rrArrayListItemH
#define rrArrayListItemH
#include "rrExporter.h"
#include "rrcStringList.h"
#include "rrArrayListItemBase.h"

//This unit contains
// 1) a base class for ArrayListItems
// 2) Template for basic type ArrayListItems, such as int, char double etc.
// 3) An ArrayListItem class, that represent an ArrayList, within an ArrayList object

namespace rrc
{

/**
 * @internal
 * a proprietaty collection class that is theoretically deprecated, but used extensively in rrc_api.cpp
 */
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
