#pragma hdrstop
#include "telAutoUtils.h"
//---------------------------------------------------------------------------

namespace telauto
{

using namespace rr;
StringList getRecordsAsStrings(const vector<rr::SelectionRecord>& _sels)
{
    StringList sels;
    for(int i = 0; i < _sels.size(); i++)
    {
        sels.add(_sels[i].to_string());
    }
    return sels;
}

}
