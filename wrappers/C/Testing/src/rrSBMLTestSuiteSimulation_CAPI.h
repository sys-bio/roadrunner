#ifndef rrSBMLTestSuiteSimulation_CAPIH
#define rrSBMLTestSuiteSimulation_CAPIH
#include "rrTestSuiteModelSimulation.h"
#include "rrc_api.h"

bool RunTest(const string& version, int number);

class SBMLTestSuiteSimulation_CAPI : public rr::TestSuiteModelSimulation
{
public:
    SBMLTestSuiteSimulation_CAPI(const string& dataOutputFolder = "",
            const string& modelFilePath = "", const string& modelFileName = "");

    ~SBMLTestSuiteSimulation_CAPI();

    void UseHandle(rrc::RRHandle handle);

    /**
     * Use current file information to load sbml from file
     */
    bool LoadSBMLFromFile();
    bool LoadSettings(const string& fName = "");
    bool Simulate();
    bool SaveResult();
    rr::RoadRunnerData GetResult();

private:

    // we don't own this ptr.
    rrc::RRHandle mRRHandle;
};

#endif
