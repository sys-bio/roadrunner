#pragma hdrstop
#include <sstream>
#include "rrLogger.h"
#include "rrRoadRunner.h"
#include "telTelluriumData.h"
#include "telAutoPlugin.h"
#include "rrUtils.h"
#include "auto2000_doc.h"
//---------------------------------------------------------------------------
using namespace std;
using namespace rr;
using namespace telauto;

AutoPlugin::AutoPlugin()
:
CPPPlugin("Auto-2000", "Bifurcation", NULL, NULL),

mRRAuto(NULL),
mAutoWorker(*this),
//PROPERTIES
//                                  value,                  name,                                   hint,                                                           descr,              alias,              readonly

mTempFolder(                        ".",                    "TempFolder",                           "Tempfolder used by auto and roadrunner"),
mKeepTempFiles(                     false,                  "KeepTempFiles",                        "Keep or remove temporary files."),
mSBML(                              "<none>",               "SBML",                                 "SBML, i.e. the model to be used to analyze"),
mScanDirection(                     "Negative",             "ScanDirection",                        "Direction of parameter scan (Positive or Negative)"),
mPreSimulation(						false,					"PreSimulation",						"Simulate system prior to steady-state analysis"),
mPreSimulationDuration(				5.0,					"PreSimulationDuration",				"Set duration for pre-simulation."),
mPreSimulationStart(				0.0,					"PreSimulationStart",					"Set start time for pre-simulation."),
mPreSimulationSteps(				100,					"PreSimulationSteps",					"Set the resolution for pre-simulation."),

//Output
mCaptureOutputFiles(                false,                  "CaptureOutputFiles",                   "Set flag to true to capture all of Autos output",              "",                 ""),
mBifurcationPoints(                 vector<int>(0),         "BifurcationPoints",                    "Points that AUTO marked as 'special'",                         "",                 ""),
mBifurcationLabels(                 StringList(""),         "BifurcationLabels",                    "Bifurcation labels for the data.",                             "",                 ""),
mBifurcationData(                   TelluriumData(),        "BifurcationData",                      "Parsed Bifurcation data.",                                     "",                 ""),
mAllowConservedMoiety(              true,                   "AllowConservedMoiety",                 "Allow conserved moiety conversion."                            "",                 ""),
mFort2(                             "<none>",               "fort2",                                "fort2",                                                        "",                 ""),
mFort3(                             "<none>",               "fort3",                                "fort3",                                                        "",                 ""),
mFort6(                             "<none>",               "BifurcationSummary",                   "BifurcationSummary (auto output file fort.6)",                 "",                 "BifurcationSummary"),
mFort7(                             "<none>",               "BifurcationDiagram",                   "BifurcationDiagram (auto output file fort.7)",                 "",                 "BifurcationDiagram"),
mFort8(                             "<none>",               "fort8",                                "fort9",                                                        "",                 ""),
mFort9(                             "<none>",               "fort9",                                "fort9",                                                        "",                 ""),
//mCalculateSteadyState(              true,                   "CalculateSteadyStateOnInit",                                "fort9",                                                        "",                 ""),
//Pure AUTO parameters
mPrincipalContinuationParameter(    "<none>",               "PrincipalContinuationParameter",       "Principal Continuation Parameter"),

//                                  value,                  name,                                   hint,                                                           descr,              alias,              readonly
mNDIM(                              1,                      "NDIM",                                 "Dimension of the system of equations",                            "",                 ""),
mIPS(                               1,                      "IPS",                                  "Constant defining the problem type",                              "",                 ""),
mIRS(                               0,                      "IRS",                                  "Restart label",                                                   "",                 ""),
mILP(                               1,                      "ILP",                                  "Locate limit points",                                             "",                 ""),
mNICP(                              1,                      "NICP",                                 "Number of free parameters",                                       "",                 ""),
mICP(                               vector<int>(0),         "ICP",                                  "Free parameters",                                                 "",                 ""),
mNTST(                              15,                     "NTST",                                 "Number of mesh intervals",                                        "",                 ""),
mNCOL(                              3,                      "NCOL",                                 "Number of collocation points per mesh interval",                  "",                 ""),
mIAD(                               3,                      "IAD",                                  "Mesh adaption",                                                   "",                 ""),
mISP(                               1,                      "ISP",                                  "Detection of branch points",                                      "",                 ""),
mISW(                               1,                      "ISW",                                  "At a restart solution",                                           "",                 ""),
mIPLT(                              0,                      "IPLT",                                 "The \"principal solution measure\"",                              "",                 ""),
mNBC(                               0,                      "NBC",                                  "Number of boundary conditions",                                   "",                 ""),
mNINT(                              0,                      "NINT",                                 "Number of integral constraints",                                  "",                 ""),
mNMX(                               1000,                   "NMX",                                  "Maximum number of continuation step along any branch",            "",                 ""),
mRL0(                               0.01,                   "RL0",                                  "Principal Continuation Parameter Lower Bound",                    "",                 "PCPLowerBound"              ),
mRL1(                               30,                     "RL1",                                  "Principal Continuation Parameter Upper Bound",                    "",                 "PCPUpperBound"              ),
mA0(                                0,                      "A0",                                   "Bifurcation diagram lower limit for the \"principal solution measure\"",  "",         ""),
mA1(                                10000,                  "A1",                                   "Bifurcation diagram upper limit for the \"principal solution measure\"",  "",         ""),
mNPR(                               50,                     "NPR",                                  "Save the solution in the solution file every NPR continuation steps",                 "",         ""),
mMXBF(                              -1,                     "MXBF",                                 "Maximum number of branch switches",                               "",         ""),
mIID(                               0,                      "IID",                                  "Controls the diagnostic file output",                             "",         ""),
mITMX(                              8,                      "ITMX",                                 "Maximum number of iterations for detecting special solutions",  "",         ""),
mITNW(                              5,                      "ITNW",                                 "Maximum number of Newton+Chord iterations",                       "",         ""),
mNWTN(                              3,                      "NWTN",                                 "Maximum number of Newton iterations",                             "",         ""),
mJAC(                               0,                      "JAC",                                  "Jacobian supplied ?",                                             "",         ""),
mEPSL(                              1e-8,                   "EPSL",                                 "Newton/Chord convergence tolerance for free parameters",          "",         ""),
mEPSU(                              1e-8,                   "EPSU",                                 "Newton/Chord convergence tolerance for solutions",                "",         ""),
mEPSS(                              1e-6,                   "EPSS",                                 "Continuation step size tolerance when locating special solutions","",         ""),
mDS(                                0.001,                  "DS",                                   "Nominal continuation step size",                                  "",         ""),
mDSMIN(                             1e-5,                   "DSMIN",                                "Minimum continuation step size",                                  "",         ""),
mDSMAX(                             0.1,                    "DSMAX",                                "Maximum continuation step size",                                  "",         ""),
mIADS(                              1,                      "IADS",                                 "Adapt the continuation step size every IADS steps",               "",         ""),
mNTHL(                              0,                      "NTHL",                                 "The number of modified parameter \"weights\" (for BVP)",          "",         ""),
mTHL(                               vector<int>(0),         "THL",                                  "Parameter index, parameter weight",                               "",         ""),
mNTHU(                              0,                      "NTHU",                                 "The number of modified solution component \"weights\" (for BVP)", "",         ""),
mTHU(                               vector<int>(0),         "THU",                                  "Component index, component weight",                               "",         ""),
mNUZR(                              0,                      "NUZR",                                 "The number of user output points specified",                    "",         ""),
mUZR(                               vector<int>(0),         "UZR",                                  "Parameter index, parameter value",                                "",         "")
{
    mVersion = "1.0.0";
    //Setup the plugin properties
    addProperties();

    //Create a roadrunner to use
    mRR = new RoadRunner;
    mRRAuto.assignRoadRunner(mRR);

    mHint ="Bifurcation Analyis using AUTO2000";
    mDescription="The Auto2000 plugin is a wrapper around the AUTO 2000 Bifurcation analysis library. This plugin was inspired and are using many of Frank Bergmann's \
ideas on how to create a usable interface to the AUTO 2000 library.";

    //The function below assigns property descriptions
    assignPropertyDescriptions();
}

AutoPlugin::~AutoPlugin()
{
    delete mRR;
}

AutoTellurimInterface& AutoPlugin::getRRAuto()
{
    return mRRAuto;
}

void AutoPlugin::setScanDirection(ScanDirection dir)
{
    if(dir == sdPositive)
    {
        mScanDirection.setValue("Positive");
    }
    else
    {
        mScanDirection.setValue("Negative");
    }
}

string AutoPlugin::getConstants()
{
    return mRRAuto.getConstantsAsString();
}

bool AutoPlugin::isWorking() const
{
    return mAutoWorker.isRunning();
}

bool AutoPlugin::resetPlugin()
{
    if(mAutoWorker.isRunning())
    {
        return false;
    }

    return true;
}

string AutoPlugin::getTempFolder()
{
    return mTempFolder.getValue();
}

string AutoPlugin::getSBML()
{
    return mSBML.getValue();
}

string AutoPlugin::getResult()
{
    stringstream msg;
    msg<<"AUTO 2000 DATA\n";
    msg<<mFort7.getValue();
    return msg.str();
}

bool AutoPlugin::execute(bool inThread)
{
    Log(lInfo)<<"Executing the AutoPlugin plugin";

    //Tempfolder setup
    if(getTempFolder() == ".")
    {
        mRR->setTempDir(getCWD());
        mRRAuto.setTempFolder(getCWD());
    }
    else
    {
        mRR->setTempDir(getTempFolder());
        mRRAuto.setTempFolder(getTempFolder());
    }


    mBifurcationData.clearValue();
    mBifurcationPoints.setValue(vector<int>(0));
    mBifurcationLabels.clearValue();
    //go away and carry out the work
    mAutoWorker.start(inThread);
    return true;
}

void AutoPlugin::addProperties()
{
    mProperties.add(&mTempFolder);
    mProperties.add(&mKeepTempFiles);
    mProperties.add(&mSBML);
    mProperties.add(&mScanDirection);
	mProperties.add(&mPreSimulation);
	mProperties.add(&mPreSimulationDuration);
	mProperties.add(&mPreSimulationStart);
	mProperties.add(&mPreSimulationSteps);
    mProperties.add(&mPrincipalContinuationParameter);
    mProperties.add(&mFort2);
    mProperties.add(&mFort3);
    mProperties.add(&mFort6);
    mProperties.add(&mFort7);
    mProperties.add(&mFort8);
    mProperties.add(&mFort9);
    mProperties.add(&mBifurcationPoints);
    mProperties.add(&mBifurcationLabels);
    mProperties.add(&mBifurcationData);
    mProperties.add(&mAllowConservedMoiety);

    //AUTO parameters
    mProperties.add(&mNDIM);
    mProperties.add(&mIPS);
    mProperties.add(&mIRS);
    mProperties.add(&mILP);
    mProperties.add(&mNICP);
    mProperties.add(&mICP);
    mProperties.add(&mNTST);
    mProperties.add(&mNCOL);
    mProperties.add(&mIAD);
    mProperties.add(&mISP);
    mProperties.add(&mISW);
    mProperties.add(&mIPLT);
    mProperties.add(&mNBC);
    mProperties.add(&mNINT);
    mProperties.add(&mNMX);
    mProperties.add(&mRL0);
    mProperties.add(&mRL1);
    mProperties.add(&mA0);
    mProperties.add(&mA1);
    mProperties.add(&mNPR);
    mProperties.add(&mMXBF);
    mProperties.add(&mIID);
    mProperties.add(&mITMX);
    mProperties.add(&mITNW);
    mProperties.add(&mNWTN);
    mProperties.add(&mJAC);
    mProperties.add(&mEPSL);
    mProperties.add(&mEPSU);
    mProperties.add(&mEPSS);
    mProperties.add(&mDS);
    mProperties.add(&mDSMIN);
    mProperties.add(&mDSMAX);
    mProperties.add(&mIADS);
    mProperties.add(&mNTHL);
    mProperties.add(&mTHL);
    mProperties.add(&mNTHU);
    mProperties.add(&mTHU);
    mProperties.add(&mNUZR);
    mProperties.add(&mUZR);
}


unsigned char* AutoPlugin::getManualAsPDF() const
{
    return pdf_doc;
}

unsigned int AutoPlugin::getPDFManualByteSize()
{
    return sizeofPDF;
}


//Functions allowing the plugin to be loaded by plugin manager
AutoPlugin* createPlugin()
{
    //allocate a new object and return it
    return new AutoPlugin;
}

const char* getImplementationLanguage()
{
    return "CPP";
}

void assignDescription(PropertyBase& para, stringstream& s)
{
    para.setDescription(s.str());
    s.str("");
}

void AutoPlugin::assignPropertyDescriptions()
{
    stringstream s;

s << "SBML document as a string. Model to be used by AUTO.";
assignDescription(mSBML, s);

s.str("Tempfolder used by auto and the plugin for saving temporary files.");
assignDescription(mTempFolder, s);

s << "Boolean indicating if temporary files should be deleted after an AUTO session or not. ";
assignDescription(mKeepTempFiles, s);

s << "Parameter instructing AUTO how to sweep its principal continuation parameter. Possible values: 'Positive', 'Negative'";
assignDescription(mScanDirection, s);

s << "Instructs bifurcation plugin to simulate model prior to attempting to estimate steady-state species concentrations.";
assignDescription(mPreSimulation, s);

s << "Explictly define the duration for the pre-simulation.";
assignDescription(mPreSimulationDuration, s);

s << "Explictly define the start time for the pre-simulation.";
assignDescription(mPreSimulationStart, s);

s << "Explicity define the number of time steps for the pre-simulation.";
assignDescription(mPreSimulationSteps, s);

s << "The principal continuation parameter (PCP) is the first parameter that AUTO will sweep. Currently only one parameter is \
supported, which then by default is the PCP.";
assignDescription(mPrincipalContinuationParameter, s);

s << "Property containing the content of the AUTO temporary file, fort.2. Fort.2 is the input file for AUTO and created by the plugin.";
assignDescription(mFort2, s);

s << "Property containing the content of the AUTO temporary file, fort.3. The content of fort.3 file is undocumented in AUTO's documentation.";
assignDescription(mFort3, s);

s << "Property containing the content of the AUTO temporary file, fort.6. The content of fort.6 file is a bifurcation session summary.";
assignDescription(mFort6, s);

s << "Property containing the content of the AUTO temporary file, fort.7. The content of fort.7 file is a bifurcation diagram on success.";
assignDescription(mFort7, s);

s << "Property containing the content of the AUTO temporary file, fort.8. The content of fort.8 file contain various statistics from the session.";
assignDescription(mFort8, s);

s << "Property containing the content of the AUTO temporary file, fort.8. Diagnostic messages, convergence history, eigenvalues, and Floquet multipliers \
are written in fort.9";
assignDescription(mFort9, s);

s << "This integer vector holds the exact point number (in the sequence of all output data) for an AUTO solution point. It can be used together with the labels in the bifurcationlabels property to asssist in \
plotting a bifurcation diagram.";
assignDescription(mBifurcationPoints, s);

s << "The bifurcation labels stinglist holds the AUTO designated solution type label for a solution point, as found in the bifurcationpoints property. Consult the AUTO documentation for\
possible label types and their meaning.";
assignDescription(mBifurcationLabels, s);

s << "The Tellurium type property, BifurcationData holds the bifurcation diagram after a session. First column is the values of the selected parameter, and succesive columns are selected species.";
assignDescription(mBifurcationData, s);

s << "Instructs the plugin to perform conserved moiety conversion. Note that if turned off, plugin might fail to return proper output.";
assignDescription(mAllowConservedMoiety, s);

s << "The NDIM property correspond to the dimension of the system of equations.";
assignDescription(mNDIM, s);

s << "Constant defining the problem type (-2 -> 14) : 1 : stationary states 2 : periodic solutions 4 : BVP";
assignDescription(mIPS, s);

s << "This constant sets the label of the solution where the computation is to be restarted. Typically 0. ";
assignDescription(mIRS, s);

s << "Fold detection; 1=ON, 0=OFF";
assignDescription(mILP, s);

s << "Property denoting the number of free parameters";
assignDescription(mNICP, s);

s << "Free parameters.";
assignDescription(mICP, s);

s << "The number of mesh intervalls.";
assignDescription(mNTST, s);

s << "The number of collocation points per mesh interval";
assignDescription(mNCOL, s);

        
s << "Mesh adaption every IAD steps; 0=OFF";
assignDescription(mIAD, s);

s << "Bifurcation detection; 0=OFF, 1=BP(FP), 3=BP(PO,BVP), 2=all";
assignDescription(mISP, s);

s << "Branch switching: 1=normal, -1=switch branch (BP, HB, PD), \
2=switch to two-parameter continuation (LP, BP, HB, TR) 3=switch to three-parameter continuation (BP)";
assignDescription(mISW, s);

s << "This constant allows redefinition of the principal solution measure, which is printed as the second \
(real) column in the fort.7 output-file. See AUTO manual for possible settings.";
assignDescription(mIPLT, s);

s << "Number of boundary conditions.";
assignDescription(mNBC, s);

s << "Number of integral conditions.";
assignDescription(mNINT, s);

s << "Maximum number of steps.";
assignDescription(mNMX, s);

s << "The lower bound on the principal continuation parameter.";;
assignDescription(mRL0, s);

s << "The upper bound on the principal continuation parameter.";;
assignDescription(mRL1, s);

s << "The lower bound on the principal solution measure.";
assignDescription(mA0, s);

s << "The upper bound on the principal solution measure.";
assignDescription(mA1, s);

s << "Save the solution in the solution file every NPR continuation steps.";
assignDescription(mNPR, s);

s << "Automatic branch switching for the first MXBF bifurcation \
points if IPS=0, 1";
assignDescription(mMXBF, s);

s << "Control diagnostic output; 0=none, 1=little, 2=normal, 4=extensive.";
assignDescription(mIID, s);

s << "Maximum number of iterations for locating special solutions/points.";
assignDescription(mITMX, s);

s << "Maximum # of correction steps.";
assignDescription(mITNW, s);

s << "Corrector uses full newton for NWTN steps.";
assignDescription(mNWTN, s);

s << "User defines derivatives; 0=no, 1=yes";
assignDescription(mJAC, s);

s << "Property setting the convergence criterion for parameters";
assignDescription(mEPSL, s);

s << "Property setting the convergence criterion for solution components";
assignDescription(mEPSU, s);

s << "Property setting the convergence criterion for special points";
assignDescription(mEPSS, s);

s << "Session start step size";
assignDescription(mDS, s);

s << "Minimum continuation step size";
assignDescription(mDSMIN, s);

s << "Maximum continuation step size";
assignDescription(mDSMAX, s);

s << "Step size adaption every IADS steps; 0=OFF";
assignDescription(mIADS, s);

s << "The number of modified parameter weights (for BVP)";
assignDescription(mNTHL, s);

s << "Parameter index, parameter weight (e.g., ICP(11)=0 means PAR(11) is excluded from the step size)";
assignDescription(mTHL, s);

s << "The number of modified solution component weights (for BVP)";
assignDescription(mNTHU, s);

s << "Component index, Component weight";
assignDescription(mTHU, s);

s << "The number of user output points specified";
assignDescription(mNUZR, s);

s << "Parameter index, parameter value (if I is negative the continuation stops at the parameter value)";
assignDescription(mUZR, s);


}



