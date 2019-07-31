#pragma hdrstop
#include <typeinfo>
#include "rrStringList.h"
#include "rrStringBuilder.h"
#include "rrStringUtils.h"
#include "telAutoConstants.h"
#include "rrUtils.h"
//---------------------------------------------------------------------------
namespace telauto
{
    using namespace rr;
    using namespace std;

string getWhiteSpaces(string& line, int maxLength);


AutoConstants::AutoConstants()
:
A0(0),
A1(10000),
DS(0.001),
DSMAX(0.1),
DSMIN(1e-5),
EPSL(1e-8),
EPSS(1e-6),
EPSU(1e-8),
IAD(3),
IADS(1),
ICP(0),
IID(0),
ILP(1),
IPLT(0),
IPS(1),
IRS(0),
ISP(1),
ISW(1),
ITMX(8),
ITNW(5),
JAC(0),
MXBF(-1),
NBC(0 ),
NCOL(3),
NDIM(1),
NICP(1),
NINT(0),
NMX(1000),
NPR(50),
NTHL(0),
NTHU(0),
NTST(15),
NUZR(0),
NWTN(3),
RL0(0.01),
RL1(30),
THL(NTHL),
THU(NTHU),
UZR(NUZR),
PreSimulation(false),
PreSimulationStart(0.0),		// By default RoadRunner simulates a system from t=0 to t=5 with 100 steps.
PreSimulationDuration(5.0),		
PreSimulationSteps(100)
{}

void AutoConstants::reset()
{}

void AutoConstants::populateFrom(Properties* props)
{
    if(!props)
    {
        throw(Exception("Handed  NULL object to function AutoConstants::populateFrom"));
    }

//    Property<int>* intProp = dynamic_cast< Property<int>* > (mProperties->getProperty("NDIM"));
    try
    {
        A0              =           dynamic_cast< Property<double>*         > (props->getProperty("A0"))->getValue();
        A1              =           dynamic_cast< Property<double>*         > (props->getProperty("A1"))->getValue();
        DS              =           dynamic_cast< Property<double>*         > (props->getProperty("DS"))->getValue();
        DSMAX           =           dynamic_cast< Property<double>*         > (props->getProperty("DSMAX"))->getValue();
        DSMIN           =           dynamic_cast< Property<double>*         > (props->getProperty("DSMIN"))->getValue();
        EPSL            =           dynamic_cast< Property<double>*         > (props->getProperty("EPSL"))->getValue();
        EPSS            =           dynamic_cast< Property<double>*         > (props->getProperty("EPSS"))->getValue();
        EPSU            =           dynamic_cast< Property<double>*         > (props->getProperty("EPSU"))->getValue();
        IAD             =           dynamic_cast< Property<int   >*         > (props->getProperty("IAD"))->getValue();
        IADS            =           dynamic_cast< Property<int   >*         > (props->getProperty("IADS"))->getValue();
        ICP             =           dynamic_cast< Property< vector<int> >*  > (props->getProperty("ICP"))->getValue();
        IID             =           dynamic_cast< Property<int   >*         > (props->getProperty("IID"))->getValue();
        ILP             =           dynamic_cast< Property<int   >*         > (props->getProperty("ILP"))->getValue();
        IPLT            =           dynamic_cast< Property<int   >*         > (props->getProperty("IPLT"))->getValue();
        IPS             =           dynamic_cast< Property<int   >*         > (props->getProperty("IPS"))->getValue();
        IRS             =           dynamic_cast< Property<int   >*         > (props->getProperty("IRS"))->getValue();
        ISP             =           dynamic_cast< Property<int   >*         > (props->getProperty("ISP"))->getValue();
        ISW             =           dynamic_cast< Property<int   >*         > (props->getProperty("ISW"))->getValue();
        ITMX            =           dynamic_cast< Property<int   >*         > (props->getProperty("ITMX"))->getValue();
        ITNW            =           dynamic_cast< Property<int   >*         > (props->getProperty("ITNW"))->getValue();
        JAC             =           dynamic_cast< Property<int   >*         > (props->getProperty("JAC"))->getValue();
        MXBF            =           dynamic_cast< Property<int   >*         > (props->getProperty("MXBF"))->getValue();
        NBC             =           dynamic_cast< Property<int   >*         > (props->getProperty("NBC"))->getValue();
        NCOL            =           dynamic_cast< Property<int   >*         > (props->getProperty("NCOL"))->getValue();
        NDIM            =           dynamic_cast< Property<int   >*         > (props->getProperty("NDIM"))->getValue();
        NICP            =           dynamic_cast< Property<int   >*         > (props->getProperty("NICP"))->getValue();
        NINT            =           dynamic_cast< Property<int   >*         > (props->getProperty("NINT"))->getValue();
        NMX             =           dynamic_cast< Property<int   >*         > (props->getProperty("NMX"))->getValue();
        NPR             =           dynamic_cast< Property<int   >*         > (props->getProperty("NPR"))->getValue();
        NTHL            =           dynamic_cast< Property<int   >*         > (props->getProperty("NTHL"))->getValue();
        NTHU            =           dynamic_cast< Property<int   >*         > (props->getProperty("NTHU"))->getValue();
        NTST            =           dynamic_cast< Property<int   >*         > (props->getProperty("NTST"))->getValue();
        NUZR            =           dynamic_cast< Property<int   >*         > (props->getProperty("NUZR"))->getValue();
        NWTN            =           dynamic_cast< Property<int   >*         > (props->getProperty("NWTN"))->getValue();
        RL0             =           dynamic_cast< Property<double>*         > (props->getProperty("RL0"))->getValue();
        RL1             =           dynamic_cast< Property<double>*         > (props->getProperty("RL1"))->getValue();
        THL             =           dynamic_cast< Property< vector<int> >*  > (props->getProperty("THL"))->getValue();
        THU             =           dynamic_cast< Property< vector<int> >*  > (props->getProperty("THU"))->getValue();
        UZR             =           dynamic_cast< Property< vector<int> >*  > (props->getProperty("UZR"))->getValue();
		// Extended python properties for Auto plugin in Tellurium.
		PreSimulation			=			dynamic_cast< Property<bool>*			> (props->getProperty("PreSimulation"))->getValue();
		PreSimulationStart		=			dynamic_cast< Property<double>*			> (props->getProperty("PreSimulationStart"))->getValue();
		PreSimulationDuration	=			dynamic_cast< Property<double>*			> (props->getProperty("PreSimulationDuration"))->getValue();
		PreSimulationSteps		=			dynamic_cast< Property<int>*			> (props->getProperty("PreSimulationSteps"))->getValue();

    }
    catch(Exception&)
    {                   
        Log(lError) << "Failed transfer data to auto constants class";
    }
    catch(const std::bad_cast&)
    {
        Log(lError) << "Failed transfer data to auto constants class";
    }
    catch(...)
    {
        Log(lError) << "Failed transfer data to auto constants class";
    }


}


string AutoConstants::getConstantsAsString()
{
    string line1  = formatN((string)"{0} {1} {2} {3}", NDIM, IPS, IRS, ILP);
    string line2  = getICP();
    string line3  = formatN("{0} {1} {2} {3} {4} {5} {6} {7}", NTST, NCOL, IAD, ISP, ISW, IPLT, NBC, NINT);
    string line4  = formatN("{0} {1} {2} {3} {4}", NMX, RL0, RL1, A0, A1);
    string line5  = formatN("{0} {1} {2} {3} {4} {5} {6}", NPR, MXBF, IID, ITMX, ITNW, NWTN, JAC);
    string line6  = formatN("{0} {1} {2}", EPSL, EPSU, EPSS);
    string line7  = formatN("{0} {1} {2} {3}", DS, DSMIN, DSMAX, IADS);
    string line8  = formatN("{0}", NTHL);
    string line9  = formatN("{0}", NTHU);
    string line10 = formatN("{0}", NUZR);

    StringList lines;
    lines.add(line1);
    lines.add(line2);
    lines.add(line3);
    lines.add(line4);
    lines.add(line5);
    lines.add(line6);
    lines.add(line7);
    lines.add(line8);
    lines.add(line8);
    lines.add(line10);

    int maxLength = lines.getLongest().size();
    StringBuilder builder;
    builder << (line1 + getWhiteSpaces(line1, maxLength) + " NDIM, IPS, IRS, ILP") << endl;
    builder << (line2 + getWhiteSpaces(line2, maxLength) + " NICP,(ICP(I),I=1 NICP)")<< endl;
    builder << line3 << getWhiteSpaces(line3, maxLength);
    builder << " NTST,NCOL,IAD,ISP,ISW,IPLT,NBC,NINT"<< endl;
    builder << (line4 + getWhiteSpaces(line4, maxLength) + " NMX,RL0,RL1,A0,A1")<< endl;
    builder << (line5 + getWhiteSpaces(line5, maxLength) + " NPR,MXBF,IID,ITMX,ITNW,NWTN,JAC")<< endl;
    builder << (line6 + getWhiteSpaces(line6, maxLength) + " EPSL,EPSU,EPSS")<< endl;
    builder << (line7 + getWhiteSpaces(line7, maxLength) + " DS,DSMIN,DSMAX,IADS")<< endl;
    builder << (line8 + getWhiteSpaces(line8, maxLength) + " NTHL,((I,THL(I)),I=1,NTHL)")<< endl;
//    WriteTHL(builder);

    builder << (line9 + getWhiteSpaces(line9, maxLength) + " NTHU,((I,THU(I)),I=1,NTHU)") << endl;
//    WriteTHU(builder);

    builder << (line10 + getWhiteSpaces(line10, maxLength) + " NUZR,((I,PAR(I)),I=1,NUZR)") << endl;
//    WriteUZR(builder);

    builder << "\n";
    builder << "\n";

    string aStr = builder.ToString();
    return aStr;
}

string AutoConstants::getICP()
{
    StringBuilder builder;
    builder << (NICP);

    for(int i = 0; i < ICP.size(); i++)
    {
        builder << " " << i; // NOTE: Totte used to have `builder<<(" " + i);` here
    }

    if (ICP.size() == 0)
    {
        builder << " 0";
    }

    return builder.ToString();
}

string getWhiteSpaces(string& line, int maxLength)
{
    int nrSpaces =  maxLength - line.size();
    return string(nrSpaces, ' ');
}

//        public string ToInputString()
//        {
//            StringBuilder builder = new StringBuilder();
//
//
//            string line1 = string.Format("{0} {1} {2} {3}", NDIM, IPS, IRS, ILP);
//            string line2 = GetICP();
//            string line3 = string.Format("{0} {1} {2} {3} {4} {5} {6} {7}", NTST, NCOL, IAD, ISP, ISW, IPLT, NBC, NINT);
//            string line4 = string.Format("{0} {1} {2} {3} {4}", NMX, RL0, RL1, A0, A1);
//            string line5 = string.Format("{0} {1} {2} {3} {4} {5} {6}", NPR, MXBF, IID, ITMX, ITNW, NWTN, JAC);
//            string line6 = string.Format("{0} {1} {2}", EPSL, EPSU, EPSS);
//            string line7 = string.Format("{0} {1} {2} {3}", DS, DSMIN, DSMAX, IADS);
//            string line8 = string.Format("{0}", NTHL);
//            string line9 = string.Format("{0}", NTHU);
//            string line10 = string.Format("{0}", NUZR);
//
//            int maxLength = GetMaxLength(new string[] { line1, line2, line3, line4, line5, line6, line7, line8, line9, line10 });
//
//
//            builder.AppendLine(line1 + WhiteSpaces(line1, maxLength) + " NDIM, IPS, IRS, ILP");
//            builder.AppendLine(line2 + WhiteSpaces(line2, maxLength) + " NICP,(ICP(I),I=1 NICP)");
//            builder.AppendLine(line3 + WhiteSpaces(line3, maxLength) + " NTST,NCOL,IAD,ISP,ISW,IPLT,NBC,NINT");
//            builder.AppendLine(line4 + WhiteSpaces(line4, maxLength) + " NMX,RL0,RL1,A0,A1");
//            builder.AppendLine(line5 + WhiteSpaces(line5, maxLength) + " NPR,MXBF,IID,ITMX,ITNW,NWTN,JAC");
//            builder.AppendLine(line6 + WhiteSpaces(line6, maxLength) + " EPSL,EPSU,EPSS");
//            builder.AppendLine(line7 + WhiteSpaces(line7, maxLength) + " DS,DSMIN,DSMAX,IADS");
//            builder.AppendLine(line8 + WhiteSpaces(line8, maxLength) + " NTHL,((I,THL(I)),I=1,NTHL)");
//            WriteTHL(builder);
//
//            builder.AppendLine(line9 + WhiteSpaces(line9, maxLength) + " NTHU,((I,THU(I)),I=1,NTHU)");
//            WriteTHU(builder);
//
//            builder.AppendLine(line10 + WhiteSpaces(line10, maxLength) + " NUZR,((I,PAR(I)),I=1,NUZR)");
//            WriteUZR(builder);
//
//            builder.AppendLine();
//            builder.AppendLine();
//
//            return builder.ToString();
//        }
}
