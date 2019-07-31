#ifndef telAutoDataH
#define telAutoDataH
#include <ostream>
#include "telTelluriumData.h"
#include "rrLogger.h"
#include "tai_exporter.h"
#include "rrStringList.h"
#include "telProperty.h"
#include "telProperties.h"
//---------------------------------------------------------------------------

// To be trashed... use autodataparser instead

using namespace rr;
using std::ostream;

class TA_DS AutoData
{
    protected:
        string                          mBifurcationDiagram;     //This is the main resulting  data file

    public:
                                        AutoData();
                                       ~AutoData();
                                        AutoData(const AutoData& data);
        AutoData&                       operator=(const AutoData& rhs);
        string                          getBifurcationDiagram() const;
        void                            setBifurcationDiagram(const string& str);

        friend TA_DS
        ostream& operator<<(ostream& stream, const AutoData& outMe);
};

template<>
inline string Property< AutoData >::getValueAsString() const
{
    Log(lDebug2)<<"Getting AutoData value as a string is not implemented";
    return "";
}

template<>
inline void Property< AutoData >::setValueFromString(const string& val)
{
    Log(lDebug2)<<"Setting AutoData value from string is not implemented";
    return;
}

template<>
inline void Property< AutoData >::setValue(AutoData* val)
{
    mValue = *(val);
}

namespace rr{

template<>
inline std::string getPropertyType<AutoData>(const AutoData& val)
{
    return "Auto Data";
}

}
#endif
