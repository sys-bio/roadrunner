/*
 * rrRoadRunnerOptions.h
 *
 *  Created on: Sep 9, 2013
 *      Author: andy
 */

#ifndef RRROADRUNNEROPTIONS_H_
#define RRROADRUNNEROPTIONS_H_

#include "rrExporter.h"
#include "Variant.h"

#include <string>
#include <vector>

#if (__cplusplus >= 201103L) || defined(_MSC_VER)
#include <memory>
#include <unordered_map>
#define cxx11_ns std
#else
#include <tr1/memory>
#include <tr1/unordered_map>
#define cxx11_ns std::tr1
#endif

#if defined(_MSC_VER)
#include "msc_stdint.h"
#else
#include <stdint.h>
#endif

namespace rr
{

/**
 * Options for loading SBML into RoadRunner.
 *
 * Future version may add additional fields to the end of this struct,
 * this way we can maintain binary compatibility with older RoadRunner versions.
 */
struct RR_DECLSPEC LoadSBMLOptions
{
    enum ModelGeneratorOpt
    {
        /**
         * perform conservation analysis.
         *
         * This causes a re-ordering of the species, so results generated
         * with this flag enabled can not be compared index wise to results
         * generated otherwise.
         *
         * currently only implemented with the C code generating model
         */
        CONSERVED_MOIETIES               = (0x1 << 0),  // => 0x00000001

        /**
         * Should the model be recompiled?
         * The LLVM ModelGenerator maintins a hash table of currently running
         * models. If this flag is NOT set, then the generator will look to see
         * if there is already a running instance of the given model and
         * use the generated code from that one.
         *
         * If only a single instance of a model is run, there is no
         * need to cache the models, and this can safetly be enabled,
         * realizing some performance gains.
         */
        RECOMPILE                        = (0x1 << 1),  // => 0x00000010

        /**
         * If this is set, then a read-only model is generated. A read-only
         * model can be simulated, but no code is generated to set model
         * values, i.e. parameters, amounts, values, etc...
         *
         * It takes a finite amount of time to generate the model value setting
         * functions, and if they are not needed, one may see some performance
         * gains, especially in very large models.
         */
        READ_ONLY                        = (0x1 << 2),  // => 0x00000100

        /**
         * Generate accessor functions to allow changing of initial
         * conditions.
         */
        MUTABLE_INITIAL_CONDITIONS       = (0x1 << 3),   // => 0x00001000

        /**
         * GVN - This pass performs global value numbering and redundant load
         * elimination cotemporaneously.
         */
        OPTIMIZE_GVN                     = (0x1 << 4),

        /**
         * CFGSimplification - Merge basic blocks, eliminate unreachable blocks,
         * simplify terminator instructions, etc...
         */
        OPTIMIZE_CFG_SIMPLIFICATION      = (0x1 << 5),

        /**
         * InstructionCombining - Combine instructions to form fewer, simple
         * instructions. This pass does not modify the CFG, and has a tendency to make
         * instructions dead, so a subsequent DCE pass is useful.
         */
        OPTIMIZE_INSTRUCTION_COMBINING   = (0x1 << 6),

        /**
         * DeadInstElimination - This pass quickly removes trivially dead instructions
         * without modifying the CFG of the function.  It is a BasicBlockPass, so it
         * runs efficiently when queued next to other BasicBlockPass's.
         */
        OPTIMIZE_DEAD_INST_ELIMINATION   = (0x1 << 7),

        /**
         * DeadCodeElimination - This pass is more powerful than DeadInstElimination,
         * because it is worklist driven that can potentially revisit instructions when
         * their other instructions become dead, to eliminate chains of dead
         * computations.
         */
        OPTIMIZE_DEAD_CODE_ELIMINATION   = (0x1 << 8),

        /**
         * InstructionSimplifier - Remove redundant instructions.
         */
        OPTIMIZE_INSTRUCTION_SIMPLIFIER  = (0x1 << 9),

        /**
         * Use the LLVM MCJIT JIT engine.
         *
         * Defaults to false.
         *
         * The MCJIT is the new LLVM JIT engine, it is not as well tested as the
         * original JIT engine. Does NOT work on LLVM 3.1
         */
        USE_MCJIT =                       (0x1 << 10),

        /**
         * Use the GPU simulation engine.
         *
         * Defaults to false.
         *
         * The GPU sim engine generates the model evaluation code and integration
	 * code on the GPU.
         */
        USE_GPUSIM =                       (0x1 << 11)
    };

    enum LoadOpt
    {
        /**
         * Do not create a default selection list when the model is loaded.
         */
        NO_DEFAULT_SELECTIONS           = (0x1 << 0)   // => 0x00000001
    };

    /**
     * initializes the struct with the default options.
     */
    LoadSBMLOptions();

    /**
     * the version this struct
     */
    uint16_t version;

    /**
     * sizeof this struct
     */
    uint16_t size;


    uint32_t modelGeneratorOpt;

    uint32_t loadFlags;
};

/**
 * RoadRunner simulation options.
 *
 * This is the full set of options that determines how RoadRunner performs
 * a simulation of an sbml model.
 *
 * This is a superset of the values stored in a sbml test suite settings file, the
 * documentation of the fields which correspond to an sbml test suite settings was
 * taken from http://sbml.org
 */
class RR_DECLSPEC SimulateOptions
{
public:
    enum Options
    {
        /**
         * reset the model to the initial state.
         */
        RESET_MODEL             = (0x1 << 0), // => 0x00000001

        /**
         * Simulate should return a raw result matrix without
         * adding any column names.
         */
        STRUCTURED_RESULT       = (0x1 << 1), // => 0x00000010

        /**
         * Make a copy of the simulation result in Python.
         *
         * Large simulations should have this disabled.
         */
        COPY_RESULT             = (0x1 << 2), // => 0x00000100
    };

    /**
     * the list of ODE solvers RoadRunner currently supports.
     *
     * The last item, INTEGRATOR_END needs to always be the last item
     * it is not a valid integrator, just used to indicate how many
     * we have.
     */
    enum Integrator
    {
        /**
         * The default CVODE integrator from the Sundials package.
         */
        CVODE = 0,  

        /**
         * Basic Gillespie stochastic integrator. 
         */
        GILLESPIE, 

        /**
         * Basic Runge-Kutta fourth order integrator.
         */
        RK4, 

        /**
         * Basic gpu integrator
         */
        GPUSIM,

        /**
         * Always has to be at the end, this way, this value indicates
         * how many integrators we have. 
         */
        INTEGRATOR_END
    };

    /**
     * the kind of integrator
     */
    enum IntegratorType
    {
        DETERMINISTIC, STOCHASTIC
    };

    /**
     * get the type of integrator.
     */
    static IntegratorType getIntegratorType(Integrator i);

    /**
     * get the textual name of the integrator.
     */
    static std::string getIntegratorNameFromId(Integrator);

    /**
     * mape the textual name of an integrator to its
     * enumerated id.
     */
    static Integrator getIntegratorIdFromName(const std::string& name);


    enum IntegratorFlags
    {
        /**
         * Is the model a stiff system? setting this to stiff causes
         * RoadRunner to load a stiff solver which could potentially be
         * extremly slow
         */
        STIFF                   = (0x1 << 0), // => 0x00000001

        /**
         * The MULTI_STEP option tells the solver to take a series of internal steps
         * and then return the solution at the point reached by that step.
         *
         * In simulate, this option will likely be slower than normal mode,
         * but may be useful to monitor solutions as they are integrated.
         *
         * This is intended to be used in combination with the
         * IntegratorListener. It this option is set, and there is a
         * IntegratorListener set, RoadRunner::integrate will run the
         * integrator in a series of internal steps, and the listner
         * will by notified at each step.
         *
         * Highly Experimental!!!
         */
        MULTI_STEP                = (0x1 << 1), // => 0x00000010

        /**
         * Perform a variable time step simulation. This will allow the
         * integrator to best choose an adaptive time step and the resulting
         * matrix will have a non-uniform time column
         */
        VARIABLE_STEP             = (0x1 << 2) // => 0b00000100
    };

    /**
     * init with default options.
     */
    SimulateOptions();

    /**
     * Set the default flags, but the fields start, duration, absolute, relative, variables,
     * amounts and concentrations are loaded from an sbml test suite settings file.
     */
    SimulateOptions(const std::string& sbmlSettingFilePath);

    /**
     * a bitmask of the options specified in the Options enumeration.
     */
    uint32_t flags;

    /**
     * which integrator to use.
     *
     * This is exposed in python via a set of custom properties
     * which automatically change the VARIABLE_STEP bitfield to
     * corespond with the appropriate integrator.
     */
    #ifndef SWIG
    Integrator integrator;
    #endif

    /**
     * Set of options to use when configuring the integrator.
     * This is a bitfield composed of options from the IntegratorOpt enum.
     */
    uint32_t integratorFlags;

    /**
     * The number of steps at which the output is sampled. The samples are evenly spaced.
     * When a simulation system calculates the data points to record, it will typically
     * divide the duration by the number of time steps. Thus, for X steps, the output
     * will have X+1 data rows.
     */
    int steps;

    /**
     * The start time of the simulation time-series data.
     * Often this is 0, but not necessarily.
     */
    double start;

    /**
     * The duration of the simulation run, in the model's units of time.
     */
    double duration;

    /**
     * A number representing the absolute difference permitted.
     */
    double absolute;

    /**
     * A float-point number representing the relative difference permitted.
     * Defaults 0.0001
     */
    double relative;

    /**
     * The variables (in addition to time) whose values will be saved in the result.
     * These are SBML model id's. Order is significant, as this determines the order
     * of the columns in the result matrix.
     *
     * Important: if a symbol in this list refers to a species in the model,
     * then that symbol will also be listed in either the amount or concentration
     * lists below.
     *
     * NOTE:If a listed variable has two underscores in it ('__'), that variable
     * is actually present only in a submodel of the main model, from the
     * Hierarchical Model Composition package, in the format submodelID__variableID.
     * If the model is flattened, the variable will appear automatically.
     */
    std::vector<std::string> variables;

    /**
     * A list of the variable whose output in the results file is in amount
     * (not concentration) units. This list of variables must be a subset of
     * the names listed in variables.
     */
    std::vector<std::string> amounts;

    /**
     * A list of the variable whose output in the results file is in concentration
     * (not amount) units. This list of variables must be a subset of the names
     * listed in variables.
     */
    std::vector<std::string> concentrations;

    /**
     * A useer specified initial time step. If this is <=  0, the integrator
     * will attempt to determine a safe initial time stpe.
     *
     * Note, for each number of steps given to RoadRunner::simulate or RoadRunner::oneStep,
     * the internal integrator may take many many steps to reach one of the external time
     * steps. This value specifies an initial value for the internal integrator
     * time step.
     */
    double initialTimeStep;

    /**
     * Specfify the minimum time step that the internal integrator
     * will use. Uses integrator estimated value if <= 0.
     */
    double minimumTimeStep;

    /**
     * Specify the maximum time step size that the internaal integrator
     * will use. Uses integrator estimated value if <= 0.
     */
    double maximumTimeStep;

    /**
     * Specify the maximum number of steps the internal integrator will use
     * before reaching the user specified time span. Uses the integrator
     * default value if <= 0.
     */
    int maximumNumSteps;

    /**
     * set an arbitrary key
     */
    void setValue(const std::string& key, const rr::Variant& value);

    const Variant& getValue(const std::string& key) const;

    bool hasKey(const std::string& key) const;

    int deleteValue(const std::string& key);

    std::vector<std::string> getKeys() const;

    /**
     * get a description of this object, compatable with python __str__
     */
    std::string toString() const;

    /**
     * get a short descriptions of this object, compatable with python __repr__.
     */
    std::string toRepr() const;

    /**
     * Integration tolerance is very integrator specific.
     *
     * This method tweaks the tolerances so that they are tight enough for the
     * current integrator to meed the SBML test suite requirements.
     */
    void tweakTolerances();


    /**
     * set the integrator field, and also update the variable step
     * options to be appropriate for the current values.
     */
    void setIntegrator(Integrator value);

private:

    /**
     * map of string to arbitrary values
     */
    typedef cxx11_ns::unordered_map<std::string, rr::Variant> VariantMap;
    VariantMap values;
};



/**
 * A set of options that determine how the top level RoadRunner class
 * should behave.
 */
struct RR_DECLSPEC RoadRunnerOptions
{
    enum Options
    {
        /**
         * RoadRunner by default dynamically generates accessor properties
         * for all sbml symbol names on the model object when it is retrieved
         * in Python. This feature is very nice for interactive use, but
         * can slow things down. If this feature is not needed, it
         * can be disabled here.
         */
        DISABLE_PYTHON_DYNAMIC_PROPERTIES             = (0x1 << 0), // => 0x00000001
    };

    /**
     * a bitmask of the options specified in the Options enumeration.
     */
    uint32_t flags;

    /**
     * step size used for numeric Jacobian calculations.
     */
    double jacobianStepSize;

    /**
     * load default valued from config.
     */
    RoadRunnerOptions();

};


} /* namespace rr */
#endif /* RRROADRUNNEROPTIONS_H_ */
