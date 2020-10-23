#pragma hdrstop
#include "auto_utils.h"

//---------------------------------------------------------------------------
std::string toAutoLabel(int nr)
{
    switch(nr)
    {
        case 1:
            return "BP";
        case 2:
            return "LP";
        case 3:
            return "HB";
        case 4:
            return "";
        case -4:
            return "UZ";
        case 5:
            return "LP";
        case 6:
            return "BP";
        case 7:
            return "PD";
        case 8:
            return "TR";
        case 9:
            return "EP";
        case -9:
            return "MX";
        default:
            return "N/A";
    }
}


