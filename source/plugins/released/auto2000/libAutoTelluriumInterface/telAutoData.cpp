#pragma hdrstop
#include "telAutoData.h"
//---------------------------------------------------------------------------
AutoData::AutoData()
{}

AutoData::~AutoData()
{}

AutoData::AutoData(const AutoData& data)
{
}

AutoData& AutoData::operator=(const AutoData& rhs)
{
    return *this;
}


string AutoData::getBifurcationDiagram() const
{
    return mBifurcationDiagram;
}

void AutoData::setBifurcationDiagram(const string& str)
{
    mBifurcationDiagram = str;
}

ostream& operator<<(ostream& stream, const AutoData& outMe)
{
    stream<<outMe.getBifurcationDiagram();
    return stream;
}

