#pragma hdrstop
#include "telAutoDataParser.h"
#include "rrException.h"
#include "telTelluriumData.h"
#include "rrUtils.h"
#include "rrLogger.h"
#include "auto_utils.h"
//---------------------------------------------------------------------------

using namespace rr;
using namespace telauto;
AutoDataParser::AutoDataParser(const string& input)
:
mBifurcationDiagram(input)
{
    if(input.size())
    {
        if(!parse())
        {
            throw(Exception("Failed parsing AutoData"));
        }
    }
}

AutoDataParser::~AutoDataParser()
{}

void AutoDataParser::resetOutput()
{
    mBifurcationDiagram = "";
    mDataHeader.clear();
    mRawSolutionData.clear();
    mSolutionData.clear();
    mBifurcationPoints.clear();
    mBifurcationLabels.clear();
}

int AutoDataParser::getNumberOfDataPoints()
{
    return mRawSolutionData.Count();
}

int AutoDataParser::getNumberOfBifurcationPoints()
{
    return mBifurcationPoints.size();
}

string AutoDataParser::getBifurcationDiagram()
{
    return mBifurcationDiagram;
}

StringList AutoDataParser::getRawSolutionData()
{
    return mRawSolutionData;
}

TelluriumData AutoDataParser::getSolutionData()
{
    return mSolutionData;
}

vector<int> AutoDataParser::getBifurcationPoints()
{
    return mBifurcationPoints;
}

StringList AutoDataParser::getBifurcationLabels()
{
    return mBifurcationLabels;
}

StringList AutoDataParser::getDataFileHeader()
{
    return mDataHeader;
}

bool AutoDataParser::parse(const string& input)
{
    resetOutput();
    if(input.size())
    {
        mBifurcationDiagram = (input);
    }
    else if(!mBifurcationDiagram.size())
    {
        return false;
    }

    

    //Parse the bifurcation diagram and get the special points
    StringList diagram(splitString(mBifurcationDiagram,"\n"));

    for(int i = 0; i < diagram.size(); i++)
    {
        string line = trim(diagram[i]);
        //Process lines starting with a '0'
        if(line.size() && line[0] == '0')
        {
            mDataHeader.add(line);
        }

        //Process lines starting with a '1', the data
        if(line.size() && line[0] == '1')
        {
            StringList dataRecord = splitString(line, " ");
            mRawSolutionData.add(line);

            //look for special points, i.e. third record not being 0 or 4
            if(dataRecord.size() > 4)
            {
                if(     dataRecord[2] == "1" ||
                        dataRecord[2] == "2" ||
                        dataRecord[2] == "3" ||
                        dataRecord[2] == "5" ||
                        dataRecord[2] == "6" ||
                        dataRecord[2] == "7" ||
                        dataRecord[2] == "8" ||
                        dataRecord[2] == "9" ||
                        dataRecord[2] == "-9"
                  )
                {
                    //We have a special point. Parse the line and populate
                    //Bifurcation point and labels vector

                    int ptNr    = abs(toInt(dataRecord[1]));
                    string lbl  = toAutoLabel(toInt(dataRecord[2]));
                    mBifurcationPoints.push_back(ptNr);
                    mBifurcationLabels.push_back(lbl);
                }
            }
        }
    }

    //Extract the data
    int nrOfDataCols = getNrOfSolutions();
    int nrOfDataRows = mRawSolutionData.Count();
    int nrOfParas = 1;      //This may change in the future

    StringList dataHeader = StringList(getDataHeaderLine());
    int paraCol = dataHeader.indexOf("PAR(0)");
    mSolutionData.reSize(nrOfDataRows, nrOfDataCols + nrOfParas);

    for(int row = 0; row < nrOfDataRows; row++)
    {
        StringList aRow = StringList(mRawSolutionData[row]);
        for(int col = 0; col < nrOfDataCols + nrOfParas; col++)
        {
            if(col == 0)
            {
                //Extract parameter value
                double para = toDouble(aRow[paraCol]);
                mSolutionData(row, col) = para;
            }
            else
            {

                stringstream ss;
                ss<<"U("<<col<<")";
                int dataCol = dataHeader.indexOf(ss.str());
                if(dataCol > -1)
                {
                    double val = toDouble(aRow[dataCol]);
                    mSolutionData(row, col) = val;
                }
                else
                {
                    throw(Exception("Failed parsing AUTO data file"));
                }
            }
        }
    }

    //Assign header to the data
    for(int col = 0; col < nrOfDataCols + nrOfParas; col++)
    {
        if(col == 0) //Only support one parameter at the moment
        {
            mSolutionData.setColumnName(0, "PAR(0)");
        }
        else
        {

            stringstream ss;
            ss<<"U("<<col<<")";
            mSolutionData.setColumnName(col, ss.str());
        }
    }

    ////If scan direction was negative, swap the data so the parameter values are increasing
    //if(getScanDirection() == sdNegative)
    //{
    //    for(int row = 0; row < nrOfDataRows/2; row++)
    //    {
    //        mSolutionData.swapRows(row, (nrOfDataRows -1 ) - row);
    //    }
    //}

    return true;
}

ScanDirection AutoDataParser::getScanDirection()
{
    //Find line in header DS =, and check sign of number after the = sign
    for(int line = 0; line < mDataHeader.Count(); line++)
    {
        string aLine = mDataHeader[line];

        if(aLine.find("DS") != std::string::npos)
        {
            int posEqual = aLine.find("=");

            //Extract the number after equal sign
            string nrStr = aLine.substr(posEqual + 1);

            //Read first number
            double nr = extractDouble(nrStr);
            if(nr > 0)
            {
                return sdPositive;
            }
            else
            {
                return sdNegative;
            }
        }
    }

    throw(Exception("Failed getting scan direction. DataHeader is empty."));
}

int AutoDataParser::getNrOfSolutions()
{
    //Check the last line in the header for U(#), where the number of such tokens are the number of solutions
    //A solution is one data trace..
    int cnt = 0;
    string hdrStr = getDataHeaderLine();
    StringList hdr(hdrStr, " ");
    for(int i = 0; i < hdr.Count(); i++)
    {
        string token = hdr[i];
        if( token[0] == 'U')
        {
            cnt++;
        }
    }
    return cnt;
}

string AutoDataParser::getDataHeaderLine()
{
    if(mDataHeader.Count() > 0)
    {
        return mDataHeader[mDataHeader.Count() -1];
    }

    return "";    
}

