#ifndef telStringListH
#define telStringListH
#include <vector>
#include <string>
#include <cstddef>
#include "telConstants.h"
#include "telCommonExporter.h"

namespace tlp
{
using std::vector;
using std::string;
using std::ostream;

class COMMON_DECLSPEC StringList
{
    protected:
        vector<string>              mStrings;
        vector<string>::iterator    mLI;

    public:
                                    StringList();
                                    StringList(char** stringList, const int& count);
                                    StringList(const string& str, const string& delimiters = ", ");
                                    StringList(const vector<string>& strings);
                                    StringList(const StringList& cp);
//                                    StringList(rrc::RRStringArrayPtr cp);
                                   ~StringList();
        void                        reSize(int size);
        void                        insertAt(const int& index, const string& item);
        void                        append(const StringList& list);
        string                      asString(const string& delimiter = gComma) const;

        /**
         * get the size to be compatible with vector<string>
         */
        size_t                      size() const;


        size_t                      count() const;


        StringList&                 operator=(const StringList& rhs);
        StringList&                 operator=(const vector<string>& rhs);

        string&                     operator[](const size_t& index);
        const string&               operator[](const size_t& index) const;
        StringList                  operator-(const StringList& rhs);
        void                        removeAt(const int& index);
        ptrdiff_t                   find(const string& item);
        ptrdiff_t                   indexOf(const string& item);
        void                        clear();
        void                        empty();
        bool                        contains(const string& item) const;
        bool                        dontContain(const string& item) const;
        string                      getLongest();
        string                      getShortest();
        void                        add(const string& item);
        vector<string>::iterator    begin();
        vector<string>::iterator    end();
        void                        preFix(const string& fix);
        void                        postFix(const string& fix);
        COMMON_DECLSPEC
        friend ostream&             operator<<(ostream& stream, const StringList& list);

        /**
         * so we can start getting rid of this and using
         * standard vector<string>
         */
        //operator vector<string>() const { return mStrings; }
        operator const vector<string>&() const;
};



}
#endif
