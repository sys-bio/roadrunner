#ifndef telRoadRunnerListH
#define telRoadRunnerListH
#include <vector>
#include <string>
#include "telJobsExporter.h"
#include "rrConstants.h"
//#include "rrObject.h"
//---------------------------------------------------------------------------

namespace rr
{
//using namespace rr;
using std::vector;
class RoadRunner;

class JOBS_DECLSPEC RoadRunnerList //: public rrObject
{
    private:

    protected:
        vector<RoadRunner*>     mRRs;

    public:
                                RoadRunnerList(const int& nrOfRRs, const string& tempFolder = rr::gEmptyString);
        virtual                ~RoadRunnerList();
        RoadRunner*             operator[](const int& index);
        unsigned int            count();

};

}
#endif
