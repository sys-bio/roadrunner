#ifndef rrArrayListH
#define rrArrayListH
#include <vector>
#include <string>
#include <list>
#include <ostream>
#include "rrArrayListItem.h"
#include "rrc_exporter.h"
#include "rrcStringList.h"


namespace rrc
{
class StringList;
using std::vector;
using std::string;

/**
 * @internal
 * a proprietaty collection class that is theoretically deprecated, but used extensively in rrc_api.cpp
 */
class C_DECL_SPEC ArrayList
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
        unsigned int                        Count() const;
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


C_DECL_SPEC std::ostream& operator<<(std::ostream& stream, const ArrayList& list);

}
#endif
