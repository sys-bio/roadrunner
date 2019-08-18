#ifndef telLoadModelFromFileH
#define telLoadModelFromFileH
#include "telThreadPool.h"
#include "telRoadRunnerList.h"
//---------------------------------------------------------------------------
namespace rr
{

class JOBS_DECLSPEC LoadModelFromFile : public ThreadPool
{
    public:
                    LoadModelFromFile(RoadRunnerList& rrs, const string& modelFName, const int& nrThreads = 16);
                   ~LoadModelFromFile();
};

}
#endif
