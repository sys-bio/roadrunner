#ifndef telLoadModelH
#define telLoadModelH
#include <string>
#include "telThreadPool.h"
#include "telRoadRunnerList.h"
#include "telJobsExporter.h"
//---------------------------------------------------------------------------

namespace rr
{
using std::string;
//using namespace rr;

class JOBS_DECLSPEC LoadModel : public ThreadPool
{
    public:
                    LoadModel(RoadRunnerList& rrs, const string& sbml, const int& nrThreads = 16, bool reCompile = true);
                   ~LoadModel();
};

}
#endif
