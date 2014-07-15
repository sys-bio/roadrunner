#include "rrSteadyStateSolver.h"
#include "rrNLEQInterface.h"


namespace rr
{

SteadyStateSolver* rr::SteadyStateSolver::New(const SteadyStateOptions*,
        ExecutableModel* m)
{
    return new NLEQInterface(m);
}

}
