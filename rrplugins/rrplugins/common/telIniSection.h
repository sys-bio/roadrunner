#ifndef telIniSectionH
#define telIniSectionH
#include <vector>
#include <string>
#include "telCommonExporter.h"
#include "telIniKey.h"

namespace tlp
{

class IniKey;
typedef std::vector<IniKey*>            KeyList;
typedef KeyList::iterator               KeyItor;
typedef std::vector<string>             NonKeyList;
typedef NonKeyList::iterator            NonKeyItor;


/**
 * @internal
 * This class stores the definition of a section. A section contains any number
 * of keys (see IniKeys), and may or may not have a comment.
 */
class COMMON_DECLSPEC IniSection
{
    private:
        bool                mIsDirty;

    public:
        string              mName;
        string              mComment;
        KeyList             mKeys;            //vector of pointers to keys
        NonKeyList          mNonKeys;         //vector of pointers to non_keys

                            IniSection();
                            IniSection(const std::string& nameValueString, const char& sep);
                           ~IniSection();
        IniKey*             CreateKey(const string& _keyName, const string& Value = gEmptyString, const string& Comment = gEmptyString);
        IniKey*             GetKey(const int& i);
        IniKey*             GetKey(const string& keyName, bool create = false);
        size_t              KeyCount();
        size_t              NonKeyCount();
        void                Clear();
        string              GetNonKeysAsString();
        string              AsString();
};
}
#endif
