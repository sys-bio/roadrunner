#ifndef rrRoadRunnerH
#define rrRoadRunnerH

#include "rrOSSpecifics.h"
#include "rrSelectionRecord.h"
#include "rrRoadRunnerOptions.h"
#include "sbml/Species.h"
#include <thread>

#ifdef _MSC_VER
#pragma warning(disable: 26812)
#pragma warning(disable: 26451)
#endif

#include "rr-libstruct/lsMatrix.h"

#ifdef _MSC_VER
#pragma warning(disable: 26812)
#pragma warning(disable: 26451)
#endif

#include <string>
#include <vector>
#include <list>
#include <set>


namespace ls {
    class LibStructural;
}

namespace rr {

    class ModelGenerator;

    class SBMLModelSimulation;

    class ExecutableModel;

    class Integrator;

    class SteadyStateSolver;

    class SensitivitySolver;

    template<class IndexType, class DataType>
    class Matrix3D;

    static std::mutex rrMtx;

    /**
     * The main RoadRunner class.
     *
     * The RoadRunner class is responsible for loading and simulating SBML models.
     *
     * MemoryManagment: Any pointer returned by a get... method is owned by the
     * RoadRunner object and does NOT have to be deleted.
     */
    class RR_DECLSPEC RoadRunner {

    public:

        /**
         * create an empty RoadRunner object with given SBML level and version.
         */
        RoadRunner(unsigned int level = 3, unsigned int version = 2);

        /**
         * load an sbml document from anywhere.
         *
         * same arguments as load.
         *
         * If options is not null, then the RoadRunner::computeAndAssignConservationLaws
         * flag is set to whatever value is specified in the options struct.
         *
         * @param uriOrSBML: a URI, local path or sbml document contents.
         * @param options: an options struct, if null, default values are used.
         */
        RoadRunner(const std::string &uriOrSBML, const Dictionary *options = 0);

        /**
         * All three of the RoadRunner options default to the empty std::string, in this
         * case, the default values are used.
         *
         * @param compiler: If LLVM build is enabled, the compiler defaults to LLVM.
         * @param tempDir: If the old external C compiler is used, this is the
         *                    where the C files are written to.
         * @param supportCodeDir: If the old external C compiler is used, this is
         *      the location where roadrunner C include files are.
         */
        RoadRunner(const std::string &compiler, const std::string &tempDir,
                   const std::string &supportCodeDir);

        /**
        * RoadRunner copy constructor
        */
        RoadRunner(const RoadRunner &rr);

        /**
        * RoadRunner copy constructor
        */
        void operator=(const RoadRunner& rr);

        /**
         * free any memory this class allocated
         */
        virtual ~RoadRunner();

        /**
         * When there are multiple instances of RoadRunner, this is the instance id.
         */
        int getInstanceID();

        /**
         * Number of currently running RoadRunner instances.
         */
        int getInstanceCount();

        /**
         * Given an sbml document or path to an sbml document, this method moves all the local parameters
         * to global parameters.
         */
        static std::string getParamPromotedSBML(const std::string &sArg);

        /**
         * information about the current state of this object.
         */
        std::string getInfo();

        /**
         * @brief return the current time as a double
         */
        double getCurrentTime();

        /**
         * The Compiler that the ModelGenerator is using to compile / interpret sbml code.
         */
        class Compiler *getCompiler();

        /**
         * Set the name of the externa compiler to use. Some ModelGenerators may have no use
         * for this value.
         *
         * An exception is raised if the std::string is invalid.
         */
        void setCompiler(const std::string &compiler);

        /**
         * get a pointer to the integrator which is currently being used to
         * time evolve the system.
         */
        Integrator *getIntegrator();

        /**
         * get a pointer to the current steady state solver
         */
        SteadyStateSolver *getSteadyStateSolver();

        /**
         * @brief get a pointer to the current sensitivities solver
         */
        SensitivitySolver *getSensitivitySolver();

        /**
         * Get an integrator by name without switching the current integrator
         */
        Integrator *getIntegratorByName(const std::string &name);

        /**
         * Get an SteadyStateSolver by name without switching the current solver
         */
        SteadyStateSolver *getSteadyStateSolverByName(const std::string &name);

        /**
         * Get an SensitivitySolver by name without switching the current solver
         */
        SensitivitySolver *getSensitivitySolverByName(const std::string &name);

        /**
         * Make an integrator for the given name
         */
        Integrator *makeIntegrator(const std::string &name);

        /**
         * Make an SteadyStateSolver* for the given name
         */
        SteadyStateSolver *makeSteadyStateSolver(const std::string &name);

        /**
         * @brief Make SensitivitySolver for the given name
         */
        SensitivitySolver *makeSensitivitySolver(const std::string &name);

        /**
         * Return a list of the names of all existing integrators.
         */
        std::vector<std::string> getExistingIntegratorNames();

        /**
         * Return a list of the names of all existing integrators.
         */
        std::vector<std::string> getExistingSteadyStateSolverNames();

        /**
         * Return a list of the names of all existing sensitivity solvers
         */
        std::vector<std::string> getExistingSensitivitySolverNames();

        /**
         * Return a std::vector of the names of all registered integrators
         */
        static std::vector<std::string> getRegisteredIntegratorNames();

        /**
         * Return a std::vector of the names of all registered SteadyStateSolver names
         */
        static std::vector<std::string> getRegisteredSteadyStateSolverNames();

        /**
         * Return a std::vector of the names of all registered sensitivity solvers
         */
        static std::vector<std::string> getRegisteredSensitivitySolverNames();


        /**
         * @brief set the current Integrator to @param name;
         */
        void setIntegrator(const std::string &name);

        /**
         * @brief set the current SteadyStateSolver to @param name;
         * @details use getRegisteredSteadyStateSolverNames to
         * get a list of available names
         */
        void setSteadyStateSolver(const std::string &name);

        /**
         * @brief set the current SensitivitySolver to @param name;
         * @details use getRegisteredSensitivitySolverNames to
         * get a list of available sensitivity solver names
         */
        void setSensitivitySolver(const std::string &name);

        /**
         * @brief returns true if integrator @param name exists.
         * @details Use getRegisteredIntegratorNames to get a list of
         * available integrator solver names.
         */
        bool integratorExists(const std::string &name);

        /**
         * @brief returns true if SteadyStateSolver @param name exists.
         */
        bool steadyStateSolverExists(const std::string &name);

        /**
         * @brief returns true if SensitivitySolver @param name exists.
         */
        bool sensitivitySolverExists(const std::string &name);

        /**
         * Ensures all integrators and steady state solvers are registered
         */
        static void registerSolvers();

        bool isModelLoaded();

        /**
         * returns the model name if a model is loaded, empty std::string otherwise.
         */
        std::string getModelName();

        /**
         * sets the model name if a model is loaded.
         */
        void setModelName(const std::string& name);

        /**
         * returns the model id if a model is loaded, empty std::string otherwise.
         */
        std::string getModelId();

        /**
         * sets the model id if a model is loaded.
         */
        void setModelId(const std::string& id);

        /**
         * @brief Clears the currently loaded model and all associated memory
         * @details Deletes jitted code and libStruct data
         * @returns True if memory was freed, false if no model was loaded
         * in the first place
         */
        bool clearModel();

        /**
         * Carry out a single integration step using a stepsize as indicated
         * in the method call.
         *
         * @param currentTime current time
         * @param stepSize the step size
         * @param reset whether to reset the model (defaults to true).
         */
        double oneStep(double currentTime, double stepSize, bool reset = true);

        /**
        * Carry out a single internal solver integration step.
        *
        * @param currentTime current time
        * @param stepSize the step size
        * @param reset whether to reset the model (defaults to true).
        */
        double internalOneStep(double currentTime, double stepSize, bool reset = true);

        /**
         * Simulate the current SBML model.
         *
         * If options is null, then the current simulation settings (start time,
         * end time, n steps) are used. If options is not null, then the
         * current simulation settings are set to the values specified by
         * options and they are used.
         *
         * The options Dictionary may contain a large number of options, for a complete
         * list of all available options for each integrator type, @see IntegratorFactory,
         * @see SimulateOptions.
         *
         * For example, to perform a simulation from time 0 to 10 with 1000 steps, using a
         * stiff integtator, you would:
         * @code
         * RoadRunner rr = RoadRunner("someFile.xml");
         * SimulateOptions opt = rr.getSimulateOptions();
         * opt.start = 0;
         * opt.duration = 10;
         * opt.steps = 1000;
         * const DoubleMatrix *result = rr.simulate(&opt);
         * @endcode
         *
         * Similarly, options specific to a particular integrator, such as the 'seed' option
         * with the Gillespie
         * integrator, this is set via the 'setIntegrator' "integrator" key, i.e.
         * @code
         * RoadRunner rr = RoadRunner("someFile.xml");
         * rr.setIntegrator("gillespie");
         * SimulateOptions opt;
         * opt.start = 0;
         * opt.duration = 10;
         * opt.steps = 1000;
         * opt.setItem("stiff", true);
         * opt.setItem("seed", 12345);
         * const DoubleMatrix *result = rr.simulate(&opt);
         * @endcode
         * Here, the "integrator" specifies the integrator to use. The "stiff" key
         * is only used by the deterministic solvers, and it is safely ignored by the
         * stochastic solvers. Also, the "seed" sets the random seed that the integrator
         * uses. For more information about all of the available options for each integrator,
         * @see IntegratorFactory::getIntegratorOptions".
         *
         * If one wants to not store the result matrix in memory and instead write it
         * to a file during simulation, one can set the output_file option. When
         * output file is nonempty, simulate() will write to its path once every
         * Config::K_ROWS_PER_WRITE rows are generated and clear the matrix. So an
         * empty result matrix is returned, and the last simulation results are not
         * stored.
         *
         * @throws an std::exception if any options are invalid.
         * @returns a borrowed reference to a DoubleMatrix object if successful. The matrix
         * will be empty if output_file is specified and nonempty.
         */
        const ls::DoubleMatrix *simulate(const SimulateOptions *options = 0);

        /**
         * @brief simulate the model using currently set integrator
         * @param start starting time to simulate
         * @param stop what time point does the simulation end?
         * @param points how many points to output (one greater than the number of steps to take).
         */
        const ls::DoubleMatrix *simulate(double start, double stop, int points);

        /**
         * @brief simulate the model using currently set integrator
         * @param times a vector of all the time outputs desired.
         */
        const ls::DoubleMatrix *simulate(const std::vector<double> &times);

        /**
         * @brief simulate a timeseries with sensitivities from start to step with num
         * data points.
         * @details Matrix3D indexed by time. Each element of the 3D matrix is a
         * Matrix<double> with rows and columns parameters and model variables respectively.
         * The parameter k determines the kth order derivative of the sensitivity information
         * that will be returned
         * @param start starting time for time series simulation
         * @param stop last time point for time series simulation
         * @param num number of data points to simulate. Determines Z of Matrix3D.
         * @param params vector of parameters that you want sensitivity for. When empty (default), compute
         * sensitivities for all parameters vs all variables.
         * @param species vector of species to include in the results
         * Default is empty, in which case all species will be included.
         * All species are selected during solving and slicing only occurs
         * at the end.
         * @param k (default 0) return the kth other derivative of the sensitivity data.
         */
        Matrix3D<double, double> timeSeriesSensitivities(
                double start, double stop, int num,
                std::vector<std::string> params = std::vector<std::string>(),
                std::vector<std::string> species = std::vector<std::string>(),
                int k = 0);

        /**
         * @brief similar to saveStateS but save data to file 
         * @param filename the filename to save the data to.
         * @param opt either 'b' (default) or 'r'. The latter is used for debugging.
         * @see RoadRunner::saveStateS
         * @see RoadRunner::loadState
         */
        void saveState(std::string filename, char opt = 'b');

        /**
         * @brief save state as binary to a stringstream so it can be loaded again later.
         * @returns new reference to a stringstream - the caller is responsible for reclaiming memory
         *   Importantly, if the stream is read into a new RoadRunner instance via
         *   RoadRunner::loadStateS then the user no longer needs to handle the reference manually
         *   because RoadRunner::loadStateS does it after loading the state.
         * @param opt either 'b' (default) or 'r'. The latter is used for debugging.
         * @see RoadRunner::loadStateS
         * @see RoadRunner::saveState
         */
        std::stringstream* saveStateS(char opt = 'b');

         /**
         * @brief Loads a roadrunner instance saved by
         * saveState with the 'b' option
         * @see RoadRunner::loadStateS
         * @see RoadRunner::saveState
         */
        void loadState(const std::string& filename);

        /**
         * @brief load state from a @param state stringstream
         * that was produced by RoadRunner::saveStateS.
         * @details The stingstream pointer should be heap allocated
         * and generated from RoadRunner::saveStateS. It is an error to
         * use a stringstream that was generated in any other way. The memory
         * associated with the stringstream is automatically cleaned up
         * after loading the state.
         */
        void loadStateS(std::stringstream* state) ;

        /**
         * RoadRunner keeps a copy of the simulation data around until the
         * next call to simulate. This matrix can be obtained here.
         */
        const ls::DoubleMatrix *getSimulationData() const;

        void setSimulateOptions(const SimulateOptions &settings);

        /**
         * get a reference to the SimulateOptions that were set either
         * by setSimulateOptions or simulate.
         */
        SimulateOptions &getSimulateOptions();

        /**
         * Get a reference to the options that determine how this class should behave.
         *
         * These are general options. For options specific for loading or simulationg,
         * @see getSimulateOptions.
         */
        RoadRunnerOptions &getOptions();


        void setOptions(const RoadRunnerOptions &);

        /**
         * get the originally loaded sbml document as a std::string.
         *
         * This may optionally up or downconvert the document to a different version, if
         * the level and version arguments are non-zero.
         *
         * If both arguments are zero, then the document is left alone and the
         */
        std::string getSBML(int level = 0, int version = 0);

        /**
         * Returns the SBML with the current model parameters. This is different than
         * getSBML which returns the orginal SBML.
         *
         * This may optionally up or downconvert the document to a different version, if
         * the level and version arguments are non-zero.
         *
         * If both arguments are zero, then the document is left alone and the original
         * version is returned.
         */
        std::string getCurrentSBML(int level = 0, int version = 0);

        /**
         * Picks up default options from config.
         *
         * The current default is
         * SelectionRecord::TIME | SelectionRecord::RATE | SelectionRecord::FLOATING.
         */
        void reset();

        /**
         * reset the model accordign to a bitfield specified by the
         * SelectionRecord::SelectionType values.
         *
         * Note, this would make more sense as an unsigned, however SWIG has issues mapping
         * an enum to an unsigned, but seems to std::map enums just fine to an int.
         *
         * For example, to reset the floating species, time and rate rule values:
         * @code
         * rr.reset(SelectionRecord::TIME | SelectionRecord::RATE | SelectionRecord::FLOATING);
         * @endcode
         *
         * @param options a bitmask made from the SelectionRecord::SelectionTypes values.
         */
        void reset(int options);

        /**
         * resets time course and steady state selection lists to defaults
         */
        void resetSelectionLists();

        /**
         * set the floating species initial concentrations.
         *
         * equivalent to ExecutableModel::reset, then ExecutableModel::setFloatingSpeciesConcentrations
         */
        void changeInitialConditions(const std::vector<double> &ic);

        /**
         * get a pointer to the ExecutableModel owned by the RoadRunner object.
         */
        ExecutableModel *getModel();

        /**
         * load an sbml document from anywhere.
         *
         * If options is not null, then the RoadRunner::computeAndAssignConservationLaws
         * flag is set to whatever value is specified in the options struct.
         *
         * WARNING, will no longer return a value in next version.
         *
         * @param uriOrSBML: a URI, local path or sbml document contents.
         * @param options: an options struct, if null, default values are used.
         */
        void load(const std::string &uriOrSBML,
                  const Dictionary *options = 0);


        /************************ Selection Ids Species Section ***********************/

        /**
         * create a selection record. This record can be used to select values.
         */
        rr::SelectionRecord createSelection(const std::string &str);

        /**
         * Returns the currently selected columns that will be returned by
         * calls to simulate() or simulateEx(,,).
         */
        std::vector<rr::SelectionRecord> &getSelections();

        /**
         * Creates a new selection based on the selection std::string,
         * and returns the value it queries.
         */
        double getValue(const std::string &sel);

        double getValue(const SelectionRecord &record);


        void setSelections(const std::vector<std::string> &selections);

        void setSelections(const std::vector<rr::SelectionRecord> &selections);

        /**
         * returns the values selected with SimulateOptions for the current model time / timestep")
         */
        std::vector<double> getSelectedValues();

        /**
         * populates a given list with all the ids that this class can accept.
         */
        void getIds(int types, std::list<std::string> &ids);

        /**
          * @author JKM
          * @brief Gets the ids for all independent floating species
          * @details Independent means (in this case) that the species is unique
          * up to a conserved cycle, i.e. each conserved cycle counts for only
          * one species. See Sauro, Systems Biology: Introduction to Pathway
          * Modeling, 1st ed. pp. 60.
          */
        std::vector<std::string> getIndependentFloatingSpeciesIds();

        /**
          * @author JKM
          * @brief Gets the ids for all dependent floating species
          * @details See @ref getIndependentFloatingSpeciesIds for an explanation
          * of independent vs. dependent.
          */
        std::vector<std::string> getDependentFloatingSpeciesIds();

        /**
        * @author KC
        * @brief Gets the ids for all floating species concentrations
        */
        std::vector<std::string> getFloatingSpeciesConcentrationIds();

        /**
          * @brief Gets the ids for all initial floating species concentrations
          * @details See @ref getIndependentFloatingSpeciesIds for an explanation
          * of independent vs. dependent.
          */
        std::vector<std::string> getFloatingSpeciesInitialConcentrationIds();

        /**
         * returns a bit field of the ids that this class supports.
         */
        int getSupportedIdTypes();


        /**
         * sets the value coresponding to the given selection std::string
         *
         * raises an exception in the selection std::string is invalid.
         */
        void setValue(const std::string &id, double value);

/************************ End Selection Ids Species Section *******************/

        /**
        * @author KC
        * @brief Returns the floating species amounts as a named array
        */
        ls::DoubleMatrix getFloatingSpeciesAmountsNamedArray();

        /**
         * @author KC
         * @brief Returns the floating species concentrations as a named array
         */
        ls::DoubleMatrix getFloatingSpeciesConcentrationsNamedArray();

        /**
        * @author KC
        * @brief Returns the boundary species amounts as a named array
        */
        ls::DoubleMatrix getBoundarySpeciesAmountsNamedArray();

        /**
        * @author KC
        * @brief Returns the boundary species concentrations as a named array
        */
        ls::DoubleMatrix getBoundarySpeciesConcentrationsNamedArray();

        /**
        * @author Claire
        * @brief Returns the independent floating species amounts as an array
        */
        std::vector<double> getIndependentFloatingSpeciesAmountsV();

        /**
        * @author Claire
        * @brief Returns the dependent floating species amounts as an array
        */
        std::vector<double> getDependentFloatingSpeciesAmountsV();

        /**
        * @author Claire
        * @brief Returns the independent floating species concentrations as an array
        */
        std::vector<double> getIndependentFloatingSpeciesConcentrationsV();

        /**
        * @author Claire
        * @brief Returns the dependent floating species concentrations as an array
        */
        std::vector<double> getDependentFloatingSpeciesConcentrationsV();

        /**
        * @author Claire
        * @brief Returns the independent floating species amounts as a named array
        */
        ls::DoubleMatrix getIndependentFloatingSpeciesAmountsNamedArray();

        /**
        * @author Claire
        * @brief Returns the dependent floating species amounts as a named array
        */
        ls::DoubleMatrix getDependentFloatingSpeciesAmountsNamedArray();

        /**
        * @author Claire
        * @brief Returns the independent floating concentrations amounts as a named array
        */
        ls::DoubleMatrix getIndependentFloatingSpeciesConcentrationsNamedArray();

        /**
        * @author Claire
        * @brief Returns the dependent floating concentrations amounts as a named array
        */
        ls::DoubleMatrix getDependentFloatingSpeciesConcentrationsNamedArray();

        /**
         * @author KC
         * @brief Returns the rate of change of the floating species as an array
         */
        std::vector<double> getRatesOfChange();

        /**
        * @author KC
        * @brief Returns the rate of change of the floating species as a named array
        */
        ls::DoubleMatrix getRatesOfChangeNamedArray();

        /**
        * @author KC
        * @brief Returns the rate of change of the independent floating species as an array
        */
        std::vector<double> getIndependentRatesOfChange();

        /**
        * @author KC
        * @brief Returns the rate of change of the independent floating species as a named array
        */
        ls::DoubleMatrix getIndependentRatesOfChangeNamedArray();

        /**
        * @author KC
        * @brief Returns the rate of change of the dependent floating species as an array
        */
        std::vector<double> getDependentRatesOfChange();

        /**
        * @author KC
        * @brief Returns the rate of change of the dependent floating species as a named array
        */
        ls::DoubleMatrix getDependentRatesOfChangeNamedArray();

        /**
         * compute the full Jacobian at the current operating point
         */
        ls::DoubleMatrix getFullJacobian();

        ls::DoubleMatrix getFullReorderedJacobian();

        /**
         * Compute the reduced Jacobian at the current operating point.
         * @param h The step sized used for central difference method.
         *          If negative, the default value from the config file is used.
         */
        ls::DoubleMatrix getReducedJacobian(double h = -1.0);

        /**
         * Returns the eigenvalues of the full jacobian.
         *
         * This returns a vector of Complex numbers.  In the python bindings,
         * the values are complex only if there are non-zero imaginary
         * parts of the values.
         */
        std::vector<std::complex<double>> getFullEigenValues();

        /**
         * Returns the eigenvalues of the reduced jacobian.
         *
         * This returns a vector of Complex numbers.  In the python bindings,
         * the values are complex only if there are non-zero imaginary
         * parts of the values.
         */
        std::vector<std::complex<double>> getReducedEigenValues();


        /**
         * Returns the eigenvalues of the full jacobian as a named array.
         *
         * This returns an N x 2 matrix where the first column is the
         * real values and the second is the imaginary part.  The rows
         * are labeled with the corresponding species ids, and the columns
         * are labeled 'real' and 'imaginary'.
         */
        ls::DoubleMatrix getFullEigenValuesNamedArray();

        /**
         * Returns the eigenvalues of the reduced jacobian as a named array.
         *
         * This returns an N x 2 matrix where the first column is the
         * real values and the second is the imaginary part.  The rows
         * are labeled with the corresponding species ids, and the columns
         * are labeled 'real' and 'imaginary'.
         */
        ls::DoubleMatrix getReducedEigenValuesNamedArray();


        ls::DoubleMatrix getLinkMatrix();

        /**
         * get the reduced stochiometry matrix. If conservation conversion is enabled,
         * this is the matrix that coresponds to the independent species.
         *
         * A synonym for getReducedStoichiometryMatrix().
         */
        ls::DoubleMatrix getNrMatrix();


        /**
         * Returns the K matrix (right nullspace of Nr)
         * The K matrix has the structure, [I K0]'
         */
        ls::DoubleMatrix getKMatrix();

        /**
         * get the reduced stochiometry matrix. If conservation conversion is enabled,
         * this is the matrix that coresponds to the independent species.
         *
         * A synonym for getNrMatrix().
         */
        ls::DoubleMatrix getReducedStoichiometryMatrix();

        /**
         * Get the stoichiometry matrix that coresponds to the full model, even it
         * it was converted via conservation conversion.
         */
        ls::DoubleMatrix getFullStoichiometryMatrix();

        ls::DoubleMatrix getExtendedStoichiometryMatrix();


        ls::DoubleMatrix getL0Matrix();


        ls::DoubleMatrix getConservationMatrix();

        ls::DoubleMatrix getUnscaledConcentrationControlCoefficientMatrix();

        ls::DoubleMatrix getScaledConcentrationControlCoefficientMatrix();

        ls::DoubleMatrix getUnscaledFluxControlCoefficientMatrix();

        ls::DoubleMatrix getScaledFluxControlCoefficientMatrix();


        /**
         * returns the list of floating species, but with a "eigen(...)" std::string
         * wrapped around them.
         */
        std::vector<std::string> getEigenValueIds();

        /**
         * Returns the unscaled elasticity for a named reaction with respect to a
         * named parameter
         */
        double getUnscaledParameterElasticity(const std::string &reactionName,
                                              const std::string &parameterName);


        ls::DoubleMatrix getFrequencyResponse(double startFrequency,
                                              int numberOfDecades, int numberOfPoints,
                                              const std::string &parameterName, const std::string &variableName,
                                              bool useDB, bool useHz);

        /**
         * This method turns on / off the computation and adherence to conservation laws.
         */
        void setConservedMoietyAnalysis(bool value);

        /**
         * is conservation analysis enabled. This is set
         */
        bool getConservedMoietyAnalysis();


        /**
         * getVersion plus info about dependent libs versions..
         */
        static std::string getExtendedVersionInfo();


        /**
         * @author JKM
         * @brief Set the differential step size used in MCA routines like @ref getCC
         */
        double getDiffStepSize() const;

        /**
         * @author JKM
         * @brief Set the differential step size used in MCA routines like @ref getCC
         */
        void setDiffStepSize(double val);

        /**
         * @author JKM
         * @brief Get the steady state threshold used in getCC
         * @details In the MCA routines, RoadRunner will keep trying to
         * converge to a steady state until this threshold is met
         */
        double getSteadyStateThreshold() const;

        /**
         * @author JKM
         * @brief Set the steady state threshold used in getCC
         * @details In the MCA routines, RoadRunner will keep trying to
         * converge to a steady state until this threshold is met
         */
        void setSteadyStateThreshold(double val);

        /**
         * @brief Get the flux threshold used in getCC
         * @details In the MCA routines, RoadRunner will keep trying to
         * converge to a steady state until this threshold is met
         */
        double getFluxThreshold() const;

        /**
         * @brief Set the flux threshold used in getCC
         * @details In getCC, RoadRunner will return 0.0 for flux control coefficients if the 
         */
        void setFluxThreshold(double val);

        /**
         * Get unscaled control coefficient with respect to a global parameter
         *
         * variableName must be either a reaction or floating species.
         *
         * parameterName must be eithe a global parameter, boundary species, or
         * conserved sum.
         */
        double getuCC(const std::string &variableName, const std::string &parameterName);

        /**
         * Get scaled control coefficient with respect to a global parameter
         *
         * The variableName must be either a reaction id, or a floating species id.
         *
         * The parameterName must be either a global parameter, boundary species,
         * or conserved sum.
         */
        double getCC(const std::string &variableName, const std::string &parameterName);

        /**
         * Get unscaled elasticity coefficient with respect to a global parameter or species
         */
        double getuEE(const std::string &reactionName, const std::string &parameterName);

        /**
         * Get unscaled elasticity coefficient with respect to a global parameter or species.
         * Optionally the model is brought to steady state after the computation.
         */
        double getuEE(const std::string &reactionName, const std::string &parameterName,
                      bool computeSteadystate);

        /**
         * Get scaled elasticity coefficient with respect to a global parameter or species
         */
        double getEE(const std::string &reactionName, const std::string &parameterName);

        /**
         * Get scaled elasticity coefficient with respect to a global parameter or species.
         * Optionally the model is brought to steady state after the computation.
         */
        double getEE(const std::string &reactionName, const std::string &parameterName,
                     bool computeSteadyState);

        /**
         * Compute the unscaled species elasticity matrix at the current operating point
         */
        ls::DoubleMatrix getUnscaledElasticityMatrix();

        /**
         * Compute the unscaled elasticity matrix at the current operating point
         */
        ls::DoubleMatrix getScaledElasticityMatrix();

        /**
         * Compute the scaled elasticity for a given reaction and given species
         */
        double getScaledFloatingSpeciesElasticity(const std::string &reactionName,
                                                  const std::string &speciesName);

        /**
         * Get a single species elasticity value
         * IMPORTANT:
         * Assumes that the reaction rates have been precomputed at the operating point !!
         */
        double getUnscaledSpeciesElasticity(int reactionId, int speciesIndex);

        /**
         * Add a species to the current model and set its concentration.
         * @param sid: the ID of the species to be added
         * @param compartment: the compartment of the species to be added
         * @param initConcentration: the initial concentration of the species to be added
         * @param hasOnlySubstanceUnits: the boolean value of the hasOnlySubstanceUnits attribute, indicating whether the species is to be used as an amount when encountered in formulas elsewhere in the model.
         * @param boundaryCondition: the boolean value of the boundaryCondition attribute, indicating whether the species is a boundary species.
         * @param substanceUnits: the substance unit of the species to be added
         * @param forceRegenerate: a boolean value to indicate if the model is regenerated
         *					       after this function call
         *						   default value is true to regenerate model after each call
         *                         of editing function
         *						   to save time for editing for multiple times, one could
         *					       set this flag to true only in the last call of editing
         */
        void addSpeciesConcentration(const std::string &sid, const std::string &compartment, double initConcentration,
                                     bool hasOnlySubstanceUnits = false, bool boundaryCondition = false,
                                     const std::string &substanceUnits = "", bool forceRegenerate = true);

        /**
         * Add a species to the current model.
         * @param sid: the ID of the species to be added
         * @param compartment: the compartment of the species to be added
         * @param initAmount: the initial amount of the species to be added
         * @param hasOnlySubstanceUnits: the boolean value of the hasOnlySubstanceUnits attribute, indicating whether the species is to be used as an amount when encountered in formulas elsewhere in the model.
         * @param boundaryCondition: the boolean value of the boundaryCondition attribute, indicating whether the species is a boundary species.
         * @param substanceUnits: the substance unit of the species to be added
         * @param forceRegenerate: a boolean value to indicate if the model is regenerated
         *					       after this function call
         *						   default value is true to regenerate model after each call
         *                         of editing function
         *						   to save time for editing for multiple times, one could
         *					       set this flag to true only in the last call of editing
         */
        void addSpeciesAmount(const std::string &sid, const std::string &compartment, double initAmount = 0,
                              bool hasOnlySubstanceUnits = false, bool boundaryCondition = false,
                              const std::string &substanceUnits = "", bool forceRegenerate = true);


        /*
        * Remove a species from the current model. Note that all reactions related to this species(as reactants,
        * products or modifiers will be removed as well.
        * @param sid: the ID of the species to be removed
        * @param forceRegenerate: a boolean value to indicate if the model is regenerated
        *					      after this function call
        *						  default value is true to regenerate model after each call
        *                         of editing function
        *						  to save time for editing for multiple times, one could
        *					      set this flag to true only in the last call of editing
        */
        void removeSpecies(const std::string &sid, bool forceRegenerate = true);

        /**
         * Set the boundary condition of an existing species.
         * By default, the boundary condition for a species is false.
         * @param sid: the ID of the species
         * @param boundaryCondition: boundary condition to be set
         * @param forceRegenerate: a boolean value to indicate if the model is regenerated
         *					       after this function call
         *						   default value is true to regenerate model after each call
         *                         of editing function
         *						   to save time for editing for multiple times, one could
         *					       set this flag to true only in the last call of editing
         */
        void setBoundary(const std::string &sid, bool boundaryCondition, bool forceRegenerate = true);

        /**
         * Set the hasOnlySubstanceUnits attribute for an existing species.
         * @param sid: the ID of a species
         * @param hasOnlySubstanceUnits: the value of hasOnlySubstanceUnits attribute to be set
         * @param forceRegenerate: a boolean value to indicate if the model is regenerated
         *					       after this function call
         *						   default value is true to regenerate model after each call
         *                         of editing function
         *						   to save time for editing for multiple times, one could
         *					       set this flag to true only in the last call of editing
         */
        void setHasOnlySubstanceUnits(const std::string &sid, bool hasOnlySubstanceUnits, bool forceRegenerate = true);

        /**
         * Get the hasOnlySubstanceUnits attribute for an existing species.
         * @param sid: the ID of a species
         */
        bool getHasOnlySubstanceUnits(const std::string &sid);

        /**
         * Set initial amount for an existing species. Initial amount/concentration set before will be unset.
         * @param sid: the ID of a species
         * @param initAmount: the initial amount to be set
         * @param forceRegenerate: a boolean value to indicate if the model is regenerated
         *					       after this function call
         *						   default value is true to regenerate model after each call
         *                         of editing function
         *						   to save time for editing for multiple times, one could
         *					       set this flag to true only in the last call of editing
         */
        void setInitAmount(const std::string &sid, double initAmount, bool forceRegenerate = true);


        /**
         * Set initial concentration for an existing species. Initial amount/concentration set before will be unset.
         * @param sid: the ID of a species
         * @param initConcentration: the initial concentration to be set
         * @param forceRegenerate: a boolean value to indicate if the model is regenerated
         *					       after this function call
         *						   default value is true to regenerate model after each call
         *                         of editing function
         *						   to save time for editing for multiple times, one could
         *					       set this flag to true only in the last call of editing
         */
        void setInitConcentration(const std::string &sid, double initConcentration, bool forceRegenerate = true);


        /**
         * Set the constant attribute for an existing species/ parameter/ compartment
         * By default, the constant attribute is false.
         * @param sid: the ID of a species/ parameter/ compartment
         * @param constant: the constant attribute to be set
         * @param forceRegenerate: a boolean value to indicate if the model is regenerated
         *					       after this function call
         *						   default value is true to regenerate model after each call
         *                         of editing function
         *						   to save time for editing for multiple times, one could
         *					       set this flag to true only in the last call of editing
         */
        void setConstant(const std::string &sid, bool constant, bool forceRegenerate = true);


        /*
        * Add a reaction to the current model
        * @param sbmlRep: the SBML representation (i.e. a reaction tag) describing the reaction to be added
        * @param forceRegenerate: a boolean value to indicate if the model is regenerated
        *					      after this function call
        *						  default value is true to regenerate model after each call
        *                         of editing function
        *						  to save time for editing for multiple times, one could
        *					      set this flag to true only in the last call of editing
        */
        void addReaction(const std::string &sbmlRep, bool forceRegenerate = true);


        /*
        * Add a reaction to the current model
        * By our default, the reaction is not reversible and not fast.
        * @param rid: the ID of reaction to be added
        * @param reactants: the list of reactant ID, double value could be inserted before ID as stoichiometry
                            e.g, [2S1] or [1.5S1]
        * @param products: the list of product stoichiometry and ID, double value could be inserted before ID as stoichiometry
                            e.g, [2S1] or [1.5S1]
        * @param kineticLaw: the kinetic formula of reaction to be added
        * @param forceRegenerate: a boolean value to indicate if the model is regenerated
        *					      after this function call
        *						  default value is true to regenerate model after each call
        *                         of editing function
        *						  to save time for editing for multiple times, one could
        *					      set this flag to true only in the last call of editing
        */
        void addReaction(const std::string &rid, std::vector<std::string> reactants, std::vector<std::string> products,
                         const std::string &kineticLaw, bool forceRegenerate = true);

        /**
         * Remove a reaction from the current model
         * @param rid: the ID of the reaction to be removed
         * @param deleteUnusedParameters; a boolean value to indicate whether to remove unused parameters from the model as well.
         * @param forceRegenerate: a boolean value to indicate if the model is regenerated
         *					       after this function call
         *						   default value is true to regenerate model after each call
         *                         of editing function
         *						   to save time for editing for multiple times, one could
         *					       set this flag to true only in the last call of editing
         */
        void removeReaction(const std::string &rid, bool deleteUnusedParameters = false, bool forceRegenerate = true);

        /*
        * Set the reversible attribut for an existing reaction in the current model
        * @param rid: the ID of reaction to be modified
        * @param reversible: the reversible attribute to be set
        * @param forceRegenerate: a boolean value to indicate if the model is regenerated
        *					      after this function call
        *						  default value is true to regenerate model after each call
        *                         of editing function
        *						  to save time for editing for multiple times, one could
        *					      set this flag to true only in the last call of editing
        */
        void setReversible(const std::string &rid, bool reversible, bool forceRegenerate = true);


        /*
        * Set the kinetic law for an existing reaction in the current model
        * @param rid: the ID of reaction to be modified
        * @param kineticLaw: the kinetic formular of reaction
        * @param forceRegenerate: a boolean value to indicate if the model is regenerated
        *					      after this function call
        *						  default value is true to regenerate model after each call
        *                         of editing function
        *						  to save time for editing for multiple times, one could
        *					      set this flag to true only in the last call of editing
        */
        void setKineticLaw(const std::string &rid, const std::string &kineticLaw, bool forceRegenerate = true);

        /**
        * Get the kinetic law of an existing reaction in the current model.
        * @param rid: the ID of reaction to be modified
        */
        std::string getKineticLaw(const std::string &rid);


        /**
         * Add a parameter to the current model
         * @param pid: the ID of the parameter to be added
         * @param value: the value of the parameter to be added
         * @param forceRegenerate: a boolean value to indicate if the model is regenerated
         *					       after this function call
         *						   default value is true to regenerate model after each call
         *                         of editing function
         *						   to save time for editing for multiple times, one could
         *					       set this flag to true only in the last call of editing
         */
        void addParameter(const std::string &pid, double value, bool forceRegenerate = true);

        /**
         * Remove a parameter from the current model
         * @param pid: the ID of the parameter to be removed
         * @param forceRegenerate: a boolean value to indicate if the model is regenerated
         *					       after this function call
         *						   default value is true to regenerate model after each call
         *                         of editing function
         *						   to save time for editing for multiple times, one could
         *					       set this flag to true only in the last call of editing
         */
        void removeParameter(const std::string &pid, bool forceRegenerate = true);


        /**
         * Add a compartment to the current model
         * @param cid: the ID of the compartment to be added
         * @param initVolume: the initial volume of the compartment to be added
         * @param forceRegenerate: a boolean value to indicate if the model is regenerated
         *					       after this function call
         *						   default value is true to regenerate model after each call
         *                         of editing function
         *						   to save time for editing for multiple times, one could
         *					       set this flag to true only in the last call of editing
         */
        void addCompartment(const std::string &cid, double initVolume, bool forceRegenerate = true);

        /**
         * Remove a compartment from the current model
         * @param cid: the ID of the compartment to be removed
         * @param forceRegenerate: a boolean value to indicate if the model is regenerated
         *					       after this function call
         *						   default value is true to regenerate model after each call
         *                         of editing function
         *						   to save time for editing for multiple times, one could
         *					       set this flag to true only in the last call of editing
         */
        void removeCompartment(const std::string &cid, bool forceRegenerate = true);


        /*
        * Add an assignment rule to the current model
        * @param vid: ID of variable that the new rule assigns formula to
        * @param formula: the math formula of assignment rule to be added
        * @param forceRegenerate: a boolean value to indicate if the model is regenerated
        *					      after this function call
        *						  default value is true to regenerate model after each call
        *                         of editing function
        *						  to save time for editing for multiple times, one could
        *					      set this flag to true only in the last call of editing
        */
        void addAssignmentRule(const std::string &vid, const std::string &formula, bool forceRegenerate = true);


        /*
        * Add a rate rule to the current model
        * @param vid: ID of variable that rules assigns formula to
        * @param formula: the math formula of rate rule to be added
        * @param forceRegenerate: a boolean value to indicate if the model is regenerated
        *					      after this function call
        *						  default value is true to regenerate model after each call
        *                         of editing function
        *						  to save time for editing for multiple times, one could
        *					      set this flag to true only in the last call of editing
        */
        void addRateRule(const std::string &vid, const std::string &formula, bool forceRegenerate = true);


        /**
         * Remove rules related to given variable from the current model
         * @param vid: ID of variable that rules assign formula to
         * @param useInitialValue: a boolean value to indicate whether using initial value as
                                   current value when the assignment rule of a variable is removed
                                   otherwise, it will keep the value of assignment rule as current value
         * @param forceRegenerate: a boolean value to indicate if the model is regenerated
         *					       after this function call
         *						   default value is true to regenerate model after each call
         *                         of editing function
         *						   to save time for editing for multiple times, one could
         *					       set this flag to true only in the last call of editing
         */
        void removeRules(const std::string &vid, bool useInitialValue = false, bool forceRegenerate = true);


        /*
        * Add an initial assignment to an exsiting symbol of the current model
        * @param vid: ID of symbol
        * @param formula: the math formula of the initial assignment
        * @param forceRegenerate: a boolean value to indicate if the model is regenerated
        *					      after this function call
        *						  default value is true to regenerate model after each call
        *                         of editing function
        *						  to save time for editing for multiple times, one could
        *					      set this flag to true only in the last call of editing
        */
        void addInitialAssignment(const std::string &vid, const std::string &formula, bool forceRegenerate = true);


        /**
         * Remove initial assignment for a symbol from the current model
         * @param vid: ID of the symbol
         * @param forceRegenerate: a boolean value to indicate if the model is regenerated
         *					       after this function call
         *						   default value is true to regenerate model after each call
         *                         of editing function
         *						   to save time for editing for multiple times, one could
         *					       set this flag to true only in the last call of editing
         */
        void removeInitialAssignment(const std::string &vid, bool forceRegenerate = true, bool errIfNotExist = true);


        /*
        * Add an event to the current model
        * @param eid: the ID of the event to be added
        * @param useValuesFromTriggerTime: indicate the moment at which the events assignments are to be evaluated
        * @param trigger: the math formula of event trigger
        * @param forceRegenerate: a boolean value to indicate if the model is regenerated
        *					      after this function call
        *						  default value is true to regenerate model after each call
        *                         of editing function
        *						  to save time for editing for multiple times, one could
        *					      set this flag to true only in the last call of editing
        */
        void addEvent(const std::string &eid, bool useValuesFromTriggerTime, const std::string &trigger,
                      bool forceRegenerate = true);

        /*
        * Add trigger to an existing event in the model
        * If the given event already has a trigger object, the given trigger will replace the old trigger of the event
        * By default, the peresistent attribute is false and the initial value attribute is false
        * @param eid: the ID of the event to add trigger
        * @param trigger: the math formula of event trigger
        * @param forceRegenerate: a boolean value to indicate if the model is regenerated
        *					      after this function call
        *						  default value is true to regenerate model after each call
        *                         of editing function
        *						  to save time for editing for multiple times, one could
        *					      set this flag to true only in the last call of editing
        */
        void addTrigger(const std::string &eid, const std::string &trigger, bool forceRegenerate = true);

        /*
        * Set the persistent attribute of the trigger of given event
        * @param eid: the ID of the event of the trigger
        * @param persistent: the persistent attribute to be set
        * @param forceRegenerate: a boolean value to indicate if the model is regenerated
        *					      after this function call
        *						  default value is true to regenerate model after each call
        *                         of editing function
        *						  to save time for editing for multiple times, one could
        *					      set this flag to true only in the last call of editing
        */
        void setPersistent(const std::string &eid, bool persistent, bool forceRegenerate = true);

        /*
        * Set the initial value attribute of the trigger of given event
        * @param eid: the ID of the event of the trigger
        * @param initValue: the initial value attribute to be set
        * @param forceRegenerate: a boolean value to indicate if the model is regenerated
        *					      after this function call
        *						  default value is true to regenerate model after each call
        *                         of editing function
        *						  to save time for editing for multiple times, one could
        *					      set this flag to true only in the last call of editing
        */
        void setTriggerInitialValue(const std::string &eid, bool initValue, bool forceRegenerate = true);

        /*
        * Add priority to an existing event in the model
        * If the given event already has a priority object, the given priority will replace the old priority of the event
        * @param eid: the ID of the event to add priority
        * @param priority: the math formula of event priority
        * @param forceRegenerate: a boolean value to indicate if the model is regenerated
        *					      after this function call
        *						  default value is true to regenerate model after each call
        *                         of editing function
        *						  to save time for editing for multiple times, one could
        *					      set this flag to true only in the last call of editing
        */
        void addPriority(const std::string &eid, const std::string &priority, bool forceRegenerate = true);

        /*
        * Add delay to an existing event in the model
        * If the given event already has a delay object, the given delay will replace the old delay of the event
        * @param eid: the ID of the event to add priority
        * @param delay: the math formula of event delay
        * @param forceRegenerate: a boolean value to indicate if the model is regenerated
        *					      after this function call
        *						  default value is true to regenerate model after each call
        *                         of editing function
        *						  to save time for editing for multiple times, one could
        *					      set this flag to true only in the last call of editing
        */
        void addDelay(const std::string &eid, const std::string &delay, bool forceRegenerate = true);


        /*
        * Add an event assignment to an existing event in the current model
        * @param eid: the ID of the event to add the assignment
        * @param vid: the ID of the variable to assign formula
        * @param fomula: the math formula to assign
        * @param forceRegenerate: a boolean value to indicate if the model is regenerated
        *					      after this function call
        *						  default value is true to regenerate model after each call
        *                         of editing function
        *						  to save time for editing for multiple times, one could
        *					      set this flag to true only in the last call of editing
        */
        void addEventAssignment(const std::string &eid, const std::string &vid, const std::string &formula,
                                bool forceRegenerate = true);


        /**
         * Remove event assignments for given variable from an existing event
         * @param eid: the ID of the event
         * @param vid: the ID of the vairable of the event assignments
         * @param forceRegenerate: a boolean value to indicate if the model is regenerated
         *					       after this function call
         *						   default value is true to regenerate model after each call
         *                         of editing function
         *						   to save time for editing for multiple times, one could
         *					       set this flag to true only in the last call of editing
         */
        void removeEventAssignments(const std::string &eid, const std::string &vid, bool forceRegenerate = true);

        /**
         * Remove an event from the current model
         * @param eid: the ID of the event to be removed
         * @param forceRegenerate: a boolean value to indicate if the model is regenerated
         *					       after this function call
         *						   default value is true to regenerate model after each call
         *                         of editing function
         *						   to save time for editing for multiple times, one could
         *					       set this flag to true only in the last call of editing
         */
        void removeEvent(const std::string &eid, bool forceRegenerate = true);

        /**
         * Validate the current SBML
         */
        void validateCurrentSBML();


        /******************************* Steady State Section *************************/

        double mcaSteadyState();

        /**
         * Compute the steady state of the model, returns the sum of squares of the
         * solution
         *
         * The steady state solver and whatever options it needs may be specified
         * via the given dictionary. For a list of all available steady state solvers,
         * @see SteadyStateSolverFactory.
         *
         * @param dict a pointer to a dictionary which has the steady state options.
         * May be NULL, in this case the existing options are used.
         */
        double steadyState(Dictionary *dict = 0);


        /**
         * Like @ref steadyState but returns a named array of the steady state values
         */
        ls::DoubleMatrix steadyStateNamedArray(const Dictionary *dict = 0);

        /**
         * returns the current set of steady state selections.
         */
        std::vector<rr::SelectionRecord> &getSteadyStateSelections();

        /**
         * parses the given list of strings and generates selections records
         * which will be used for the steady state selections.
         */
        void setSteadyStateSelections(const std::vector<std::string> &
        steadyStateSelections);

        /**
         * makes a copy of an existing list of selection records. These will be
         * saved and used for selection values in getSteadyStateValues().
         */
        void setSteadyStateSelections(const std::vector<rr::SelectionRecord> &
        steadyStateSelections);

        /**
         * Performs a steady state calculation (evolves the system to a steady
         * state), then calculates and returns the set of values specifed by
         * the steady state selections.
         */
        std::vector<double> getSteadyStateValues();

        /**
         * Returns a std::vector of the steady state selection strings
         */
        std::vector<std::string> getSteadyStateSelectionStrings() const;

        /**
         * Like @ref getSteadyStateValues but returns a named array
         */
        ls::DoubleMatrix getSteadyStateValuesNamedArray();

        /**
          * @brief regenerate a model, this recompiles the internal runtime model.
          * @details This would be used after a series of addSpecies, addReaction calls
          * and the model is then ready to be compiled into executable form.
         */
        void regenerateModel(bool forceRegenerate = true, bool reset = false);


        /******************************* End Steady State Section *********************/

        /*********              Used by rrplugins             *************************/

        /**
         * Alias for this function on the child model object.
         */
        void setBoundarySpeciesByIndex(const int &index, const double &value);

        /**
         * Alias for this function on the child model object.
         */
        int getNumberOfIndependentSpecies();  

        /**
         * Alias for this function on the child model object.
         */
        int getNumberOfDependentSpecies();

        /**
         * Alias for this function on the child model object.
         * use ExecutableModel::getGlobalParameterIds
         */
        std::vector<std::string> getGlobalParameterIds();

        /**
         * Alias for this function on the child model object.
         */
        std::vector<std::string> getBoundarySpeciesIds();

        /**
         * Get the Ids of the elements assigned by assignment rules.
         */
        std::vector<std::string> getAssignmentRuleIds();

        /**
         * Get the Ids of the elements assigned by rate rules.
         */
        std::vector<std::string> getRateRuleIds();

        /**
         * Get the Ids of the elements with an initial assignment.
         */
        std::vector<std::string> getInitialAssignmentIds();

        /**
        * @brief Gets the ids for all boundary species concentrations
        */
        std::vector<std::string> getBoundarySpeciesConcentrationIds();

        /**
         * Alias for this function on the child model object.
         */
        double getBoundarySpeciesByIndex(const int &index);

        /**
         * Alias for this function on the child model object.
         * use ExecutableModel::getGlobalParameterValues
         */
        double getGlobalParameterByIndex(const int &index);

        /**
         * @author ciaran welsh
         * @brief set the parameter with id @param param to @param value
         */
        void setGlobalParameterByName(const std::string &param, double value);

        /**
         * @author ciaran welsh
         * @brief get the value of global parameter with id @param param
         *
         */
        double getGlobalParameterByName(const std::string &param);



        /******** !!! DEPRECATED INTERNAL METHODS * THESE WILL BE REMOVED!!! **********/

        /**
         * @author MTK, JKM
         * @brief Returns the sum of each conserved cycle
         */
        std::vector<double> getConservedMoietyValues();

        std::vector<std::string> getConservedMoietyIds();

        /**
         * @brief Set the value of Config::RANDOM_SEED
         */
        void setSeed(long int seed, bool resetModel = true);

        /**
         * @brief Returns the value of Config::RANDOM_SEED
         */
        int64_t getSeed(const std::string &integratorName="");


        /**
         * @brief Reset seed's value
         * @details Set the value of Config::RANDOM_SEED to -1
         *
         */
        void resetSeed();

#ifndef SWIG // deprecated methods not SWIG'ed

        /**
         * Alias for this function on the child model object.
         */
        int getNumberOfReactions();

        /**
         * Alias for this function on the child model object.
         */
        double getReactionRate(const int &index);

        /**
         * Alias for this function on the child model object.
         */
        double getRateOfChange(const int &index);

        /**
         * Alias for this function on the child model object.
         */
        std::vector<std::string> getRateOfChangeIds();

        /**
         * Alias for this function on the child model object.
         */
        int getNumberOfCompartments();

        /**
         * Alias for this function on the child model object.
         */
        void setCompartmentByIndex(const int &index, const double &value);

        /**
         * Alias for this function on the child model object.
         */
        double getCompartmentByIndex(const int &index);

        /**
         * Alias for this function on the child model object.
         */
        std::vector<std::string> getCompartmentIds();

        /**
         * Alias for this function on the child model object.
         */
        int getNumberOfBoundarySpecies();

        /**
         * Alias for this function on the child model object.
         */
        void setBoundarySpeciesConcentrations(const std::vector<double> &values);

        /**
         * Alias for this function on the child model object.
         */
        void setBoundarySpeciesAmounts(const std::vector<double> &values);

        /**
         * Alias for this function on the child model object.
         */
        int getNumberOfFloatingSpecies();

        /**
         * Alias for this function on the child model object.
         */
        double getFloatingSpeciesByIndex(int index);

        /**
         * Alias for this function on the child model object.
         */
        void setFloatingSpeciesByIndex(int index, double value);

        /**
         * Alias for this function on the child model object.
         */
        std::vector<double> getFloatingSpeciesConcentrationsV();

        /**
        * Alias for this function on the child model object.
        */
        std::vector<double> getFloatingSpeciesAmountsV();

        /**
        * Alias for this function on the child model object.
        */
        std::vector<double> getBoundarySpeciesConcentrationsV();

        /**
        * Alias for this function on the child model object.
        */
        std::vector<double> getBoundarySpeciesAmountsV();

        /**
         * Alias for this function on the child model object.
         */
        std::vector<double> getFloatingSpeciesInitialConcentrations();

        /**
         * Alias for this function on the child model object.
         */
        void setFloatingSpeciesConcentrations(const std::vector<double> &values);

        /**
         * Alias for this function on the child model object.
         */
        void setFloatingSpeciesInitialConcentrationByIndex(const int &index,
                              const double &value);

        /**
         * Alias for this function on the child model object.
         */
        void setFloatingSpeciesInitialConcentrations(const std::vector<double> &values);

        /**
         * Alias for this function on the child model object.
         */
        std::vector<std::string> getFloatingSpeciesIds();

        /**
         * Alias for this function on the child model object.
         */
        std::vector<std::string> getFloatingSpeciesInitialConditionIds();

        /**
         * Alias for this function on the child model object.
         */
        size_t getNumberOfGlobalParameters();

        /**
         * Alias for this function on the child model object.
         */
        void setGlobalParameterByIndex(const int index, const double value);


        /**
         * Alias for this function on the child model object.
         * use ExecutableModel::getGlobalParameterValues
         */
        std::vector<double> getGlobalParameterValues();

        /**
         * Updates the model based on all recent changes.
         */
        void evalModel();


        /**
         * Alias for this function on the child model object.
         * use ExecutableModel::getReactionRates
         */
        std::vector<double> getReactionRates();

        /**
         * Alias for this function on the child model object.
         * returns a list of reaction ids obtained from
         * ExecutableModel::getReactionId
         */
        std::vector<std::string> getReactionIds();

       /**@cond PRIVATE */
       /**
         * @deprecated
         *
         * C backend only
         * set the location where the ModelGenerator creates temporary files, such
         * as shared libraries.
         */
        void setTempDir(const std::string &folder);

        /**
         * @deprecated
         *
         * legacy C backend only
         * get the ModelGenerator's temporary file directory.
         */
        std::string getTempDir();

        /**
         * Search for the element with the given ID, but only among SBML elements that have mathematical meaning:  Species, Parameters, Compartments, and SpeciesReferences (in that order).
         */
        static const libsbml::SBase* getElementWithMathematicalMeaning(const libsbml::Model* model, const std::string& id);
        /** @endcond PRIVATE */

#endif // #ifndef SWIG

    private:

        /**
         * True once the llvm initialization routines have been run.
         * This is necessary because they are not threadsafe.
         */
        static bool llvmInitialized;
        static bool solversRegistered;

        /**
         * private implementation class, can only access if inside
         * the implementation file.
         */
        class RoadRunnerImpl* impl;

        const int fileMagicNumber = 0xAD6F52;
        const int dataVersionNumber;

        void fixDependentSpeciesValues(int except, double *ref);

        /**
         * @brief calls llvm initialization routines in a thread safe way
         */
        void initLLVM();

        size_t createDefaultSteadyStateSelectionList();

        size_t createDefaultTimeCourseSelectionList();

        /**
         * copies the current selection values into the n'th row of the
         * given matrix
         */
        void getSelectedValues(ls::DoubleMatrix &results, int nRow,
                               double currentTime);

        /**
         * copies the current selection values into the given std::vector.
         */
        void getSelectedValues(std::vector<double> &results, double currentTime);

        bool populateResult();


        double getNthSelectedOutput(size_t index, double currentTime);

        bool isParameterUsed(const std::string &sid);

        void getAllVariables(const libsbml::ASTNode *node, std::set<std::string> &ids);

        /// Get the row index of the time variable in the output array (returns -1 if time is not selected)
        int getTimeRowIndex();

        enum VariableType {
            vtSpecies = 0, vtFlux
        };

        double getVariableValue(const VariableType variableType,
                                const int variableIndex);

        /**
         * the LibStruct is normally null, only created on demand here.
         */
        ls::LibStructural *getLibStruct();

        /**
         * If the specified integrator does not exist, create it, and point the
         * integrator pointer to it.
         */
        //void updateIntegrator();

        bool createDefaultSelectionLists();

        /**
         * creates a selection list from the amounts / conc / variables ivars of the
         * SimulationOptions struct.
         */
        size_t createTimeCourseSelectionList();

        std::vector<SelectionRecord> getSelectionList();

        /**
         * The simulateOptions may be changed at any time. This updates
         * the integrators just before they are used with the
         * potentially changed options.
         */
        void applySimulateOptions();


        enum JacobianMode {
            JACOBIAN_FULL, JACOBIAN_REDUCED
        };

        std::vector<std::complex<double> > getEigenValues(JacobianMode mode);

        ls::DoubleMatrix getEigenValuesNamedArray(JacobianMode mode);

        /*
        * Check if the id already existed in the model
        */
        void checkID(const std::string &functionName, const std::string &sid);

        /*
        * Parse a std::string with format stoichiometry + sID and return its stoichiometry value and sID
        */
        void parseSpecies(const std::string &species, double *stoichiometry, char **sid);

        /*
        * Remove a variable from the current model.
        */
        void removeVariable(const std::string &sid);

        /*
        * check recursively if a ASTnode or any of its child has the given variable
        */
        bool hasVariable(const libsbml::ASTNode *node, const std::string &sid);

        /*
        * Get the names of all the species involved in a given AST
        */
        void getSpeciesIdsFromAST(const libsbml::ASTNode *node, std::vector<std::string> &species);

        void getSpeciesIdsFromAST(const libsbml::ASTNode *node, std::vector<std::string> &species,
                                  std::vector<std::string> &speciesNames);

        void saveSelectionVector(std::ostream &, std::vector<SelectionRecord> &);

        void loadSelectionVector(std::istream &, std::vector<SelectionRecord> &);
    };

}

#endif // rrRoadRunnerH