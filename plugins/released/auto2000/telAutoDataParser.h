#ifndef telAutoDataParserH
#define telAutoDataParserH
#include <vector>
#include "telTelluriumData.h"
#include "rrStringList.h"
#include "rrConstants.h"
#include "libAutoTelluriumInterface/telAutoConstants.h"
//---------------------------------------------------------------------------

using rr::StringList;
using std::vector;

class AutoDataParser
{
    public:
                                                AutoDataParser(const string& input = rr::gEmptyString);
                                               ~AutoDataParser();
        //Data input
        bool                                    parse(const string& input = rr::gEmptyString);
        int                                     getNumberOfDataPoints();
        int                                     getNumberOfBifurcationPoints();
        string                                  getBifurcationDiagram();
        StringList                              getDataFileHeader();
        StringList                              getRawSolutionData();
        vector<int>                             getBifurcationPoints();
        StringList                              getBifurcationLabels();
        rr::TelluriumData                      getSolutionData();

        telauto::ScanDirection                  getScanDirection();

    protected:
        string                                  mBifurcationDiagram;
        rr::StringList                         mDataHeader;
        rr::StringList                         mRawSolutionData;
        rr::TelluriumData                      mSolutionData;
        vector<int>                             mBifurcationPoints;
        vector<string>                          mBifurcationLabels;
        void                                    resetOutput();
        int                                     getNrOfSolutions();
        string                                  getDataHeaderLine();

};

#endif
