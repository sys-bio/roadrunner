#ifndef telAutoUtilsH
#define telAutoUtilsH
#include <vector>
#include "telStringList.h"
//#include "rr/rrRoadRunner.h"
#include "../../../source/rrSelectionRecord.h"
#include "tai_exporter.h"
//---------------------------------------------------------------------------

namespace telauto
{

using std::vector;
/**
    \brief Takes a vector of SelectionRecords and returns their representation as
    a StringList
*/
TA_DS tlp::StringList     getRecordsAsStrings(const vector< rr::SelectionRecord >& folder);

}
#endif
