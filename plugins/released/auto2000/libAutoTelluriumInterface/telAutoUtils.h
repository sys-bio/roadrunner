#ifndef telAutoUtilsH
#define telAutoUtilsH
#include <vector>
#include "rrStringList.h"
#include "rrRoadRunner.h"
#include "tai_exporter.h"
//---------------------------------------------------------------------------

namespace telauto
{

using std::vector;
/**
    \brief Takes a vector of SelectionRecords and returns their representation as
    a StringList
*/
TA_DS rr::StringList     getRecordsAsStrings(const vector< rr::SelectionRecord >& folder);

}
#endif
