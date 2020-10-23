#ifndef telStringBuilderH
#define telStringBuilderH
#include <sstream>
#include <string>
#include "telConstants.h"
#include "telCommonExporter.h"

using std::stringstream;
using std::string;

namespace tlp
{

class COMMON_DECLSPEC StringBuilder
{
    protected:
        stringstream                 mStringing;

    public:
                                    StringBuilder(const string& aStr = gEmptyString);
        stringstream&               operator<<(const string& str);
        stringstream&               operator<<(const char& ch);
        stringstream&               operator<<(const int& ch);
        string                      ToString();

        void                        NewLine(const string& line = gEmptyString);
        void                        AppendLine(const string& line = gEmptyString);
        void                        Line(const string& line);
        void                        TLine(const string& line, const int& tabs = 1);
        void                        Clear();
};

}

#endif
