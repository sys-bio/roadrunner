#ifndef rrSteadyStateSolverH
#define rrSteadyStateSolverH
#include "rrExporter.h"
#include "Configurable.h"
#include "rrExecutableModel.h"
#include <vector>


namespace rr
{

/**
 * place holder for now.
 */
struct SteadyStateOptions
{
};

/**
 * @internal
 */
class RR_DECLSPEC SteadyStateSolver : public Configurable
{

public:

    virtual ~SteadyStateSolver() {};
    virtual double solve(const std::vector<double>& yin) = 0;


    /**
     * factory method to create a new steady state solver.
     */
    static SteadyStateSolver* New(const SteadyStateOptions*, ExecutableModel*);
};

}
#endif
