#ifndef rrRoadRunnerH
#define rrRoadRunnerH

#include "rrOSSpecifics.h"
#include "rr-libstruct/lsMatrix.h"
#include "rrSelectionRecord.h"
#include "rrRoadRunnerData.h"
#include "rrRoadRunnerOptions.h"
#include "Configurable.h"

#include <string>
#include <vector>
#include <list>

namespace ls
{
class LibStructural;
}

namespace rr
{

class ModelGenerator;
class SBMLModelSimulation;
class ExecutableModel;
class Integrator;

/**
 * The main RoadRunner class.
 *
 * The RoadRunner class is responsible for loading and simulating SBML models.
 *
 * MemoryManagment: Any pointer returned by a get... method is owned by the
 * RoadRunner object and does NOT have to be deleted.
 */
class RR_DECLSPEC RoadRunner : public Configurable
{

public:

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
    RoadRunner(const std::string& uriOrSBML = "",
            const LoadSBMLOptions* options = 0);

    /**
     * All three of the RoadRunner options default to the empty string, in this
     * case, the default values are used.
     *
     * @param compiler: If LLVM build is enabled, the compiler defaults to LLVM.
     * @param tempDir: If the old external C compiler is used, this is the
     *                    where the C files are written to.
     * @param supportCodeDir: If the old external C compiler is used, this is
     *      the location where roadrunner C include files are.
     */
    RoadRunner(const std::string& compiler, const std::string& tempDir,
            const std::string& supportCodeDir);

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
     * given an sbml document, this method moves all the local parameters
     * to global parameters.
     */
    static std::string getParamPromotedSBML(const std::string& sArg);

    /**
     * information about the current state of this object.
     */
    std::string getInfo();

    /**
     * The Compiler that the ModelGenerator is using to compile / interpret sbml code.
     */
    class Compiler* getCompiler();

    /**
     * Set the name of the externa compiler to use. Some ModelGenerators may have no use
     * for this value.
     *
     * An exception is raised if the string is invalid.
     */
    void setCompiler(const std::string& compiler);

    /**
     * get a pointer to the integrator which is currently being used to
     * time evolve the system.
     */
    Integrator *getIntegrator();

    /**
     * gets a pointer to a specific integrator.
     *
     * Throws a std::invalid_argument if intg is not valid.
     */
    Integrator *getIntegrator(SimulateOptions::Integrator intg);

    bool isModelLoaded();

    /**
     * returns the model name if a model is loaded, empty string otherwise.
     */
    std::string getModelName();

    bool unLoadModel();


    /**
     * Carry out a single integration step using a stepsize as indicated
     * in the method call.
     *
     * @param t0 starting time
     * @param tf final time
     * @param options override current options.
     */
    double integrate(double t0, double tf, const SimulateOptions* options = 0);

    /**
     * @deprecated, use integrate instead.
     */
    double oneStep(double currentTime, double stepSize, bool reset = true);

    /**
     * simulate the current SBML model.
     *
     * If options is null, then the current simulation settings (start time,
     * end time, n steps) are used. If options is not null, then the
     * current simulation settings are set to the values specified by
     * options and they are used.
     *
     * @returns a RoadRunnerData object which is owned by the the RoadRunner
     * object if successfull, 0 on failure.
     */
    const DoubleMatrix *simulate(const SimulateOptions* options = 0);

    /**
     * RoadRunner keeps a copy of the simulation data around until the
     * next call to simulate. This matrix can be obtained here.
     */
    const DoubleMatrix* getSimulationData() const;

    #ifndef SWIG // deprecated methods not SWIG'ed

    /**
     * Use getSimulationData() instead.
     * Also, can use the RoadRunnerData::RoadRunnerData(const RoadRunner*) ctor
     */
    RR_DEPRECATED(rr::RoadRunnerData *getSimulationResult());

    #endif

    void setSimulateOptions(const SimulateOptions& settings);

    /**
     * get a reference to the SimulateOptions that were set either
     * by setSimulateOptions or simulate.
     */
    SimulateOptions& getSimulateOptions();

    /**
     * Get a reference to the options that determine how this class should behave.
     *
     * These are general options. For options specific for loading or simulationg,
     * @see getSimulateOptions.
     */
    RoadRunnerOptions& getOptions();


    void setOptions(const RoadRunnerOptions&);

    /**
     * get the currently loaded sbml document as a string.
     */
    std::string getSBML();

    /**
     * Reset the simulator back to the initial conditions specified in the SBML model,
     * provided an SBML model is loaded.
     */
    void reset();

    /**
     * @internal
     * set the floating species initial concentrations.
     *
     * equivalent to ExecutableModel::reset, then ExecutableModel::setFloatingSpeciesConcentrations
     *
     * @deprecated
     */
    void changeInitialConditions(const std::vector<double>& ic);

    /**
     * get the ModelGenerator that is used to create executable (runnable) models.
     */
    ModelGenerator* getModelGenerator();

    /**
     * get a pointer to the ExecutableModel owned by the RoadRunner object.
     */
    ExecutableModel* getModel();

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
    void load(const std::string& uriOrSBML,
            const LoadSBMLOptions* options = 0);

    /**
     * creates a new xml element that represent the current state of this
     * Configurable object and all if its child objects.
     *
     * This node needs to be consumed by Configurable::xmlFromConfigNode
     */
    virtual _xmlNode *createConfigNode();

    /**
     * Given an xml element, the Configurable object should pick its needed
     * values that are stored in the element and use them to set its
     * internal configuration state.
     */
    virtual void loadConfig(const _xmlDoc* doc);

    /**
     * recurse through all of the child configurable objects that this
     * class ownes and build an assemble all of thier configuration parameters
     * into a single xml document which is returned as a string.
     *
     * The value of this result depends on what child objects are presently loaded.
     */
    std::string getConfigurationXML();

    /**
     * given a xml document, which should have been returned from getConfigurationXML,
     * this method recurses though all the child configurable elements and sets thier
     * configuration to the values specified in the document.
     */
    void setConfigurationXML(const std::string& xml);


/************************ Selection Ids Species Section ***********************/
#if (1) /**********************************************************************/
/******************************************************************************/

    /**
     * create a selection record. This record can be used to select values.
     */
    rr::SelectionRecord createSelection(const std::string& str);

    /**
     * Returns the currently selected columns that will be returned by
     * calls to simulate() or simulateEx(,,).
     */
    std::vector<rr::SelectionRecord>& getSelections();

    /**
     * Creates a new selection based on the selection string,
     * and returns the value it queries.
     */
    double getValue(const std::string& sel);

    double getValue(const SelectionRecord& record);


    void setSelections(const std::vector<std::string>& selections);

    void setSelections(const std::vector<rr::SelectionRecord>& selections);


    /**
     * returns the values selected with SimulateOptions for the current model time / timestep")
     */
    std::vector<double> getSelectedValues();

    /**
     * populates a given list with all the ids that this class can accept.
     */
    void getIds(int types, std::list<std::string> &ids);

    /**
     * returns a bit field of the ids that this class supports.
     */
    int getSupportedIdTypes();


    /**
     * sets the value coresponding to the given selection string
     *
     * raises an exception in the selection string is invalid.
     */
    void setValue(const std::string& id, double value);

/************************ End Selection Ids Species Section *******************/
#endif /***********************************************************************/
/******************************************************************************/

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
     * Returns eigenvalues, first column real part, second column imaginary part
     */
    ls::DoubleMatrix getEigenvalues();

    std::vector<Complex> getEigenvaluesCpx();

    ls::DoubleMatrix getLinkMatrix();

    /**
     * get the reduced stochiometry matrix. If conservation conversion is enabled,
     * this is the matrix that coresponds to the independent species.
     *
     * A synonym for getReducedStoichiometryMatrix().
     */
    ls::DoubleMatrix getNrMatrix();

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


    ls::DoubleMatrix getL0Matrix();


    ls::DoubleMatrix getConservationMatrix();
    ls::DoubleMatrix getUnscaledConcentrationControlCoefficientMatrix();
    ls::DoubleMatrix getScaledConcentrationControlCoefficientMatrix();
    ls::DoubleMatrix getUnscaledFluxControlCoefficientMatrix();
    ls::DoubleMatrix getScaledFluxControlCoefficientMatrix();


    /**
     * returns the list of floating species, but with a "eigen(...)" string
     * wrapped around them.
     */
    std::vector<std::string> getEigenvalueIds();

    /**
     * Returns the unscaled elasticity for a named reaction with respect to a
     * named parameter
     */
    double getUnscaledParameterElasticity(const string& reactionName,
            const string& parameterName);


    Matrix<double> getFrequencyResponse(double startFrequency,
            int numberOfDecades, int numberOfPoints,
            const string& parameterName, const string& variableName,
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
     * Returns the SBML with the current parameterset.
     */
    std::string getCurrentSBML();


    /**
     * getVersion plus info about dependent libs versions..
     */
    static std::string getExtendedVersionInfo();


    /**
     * Get unscaled control coefficient with respect to a global parameter
     *
     * variableName must be either a reaction or floating species.
     *
     * parameterName must be eithe a global parameter, boundary species, or
     * conserved sum.
     */
    double getuCC(const std::string& variableName, const std::string& parameterName);

    /**
     * Get scaled control coefficient with respect to a global parameter
     *
     * The variableName must be either a reaction id, or a floating species id.
     *
     * The parameterName must be either a global parameter, boundary species,
     * or conserved sum.
     */
    double getCC(const std::string& variableName, const std::string& parameterName);

    /**
     * Get unscaled elasticity coefficient with respect to a global parameter or species
     */
    double getuEE(const std::string& reactionName, const std::string& parameterName);

    /**
     * Get unscaled elasticity coefficient with respect to a global parameter or species.
     * Optionally the model is brought to steady state after the computation.
     */
    double getuEE(const std::string& reactionName, const std::string& parameterName,
            bool computeSteadystate);

    /**
     * Get scaled elasticity coefficient with respect to a global parameter or species
     */
    double getEE(const std::string& reactionName, const std::string& parameterName);

    /**
     * Get scaled elasticity coefficient with respect to a global parameter or species.
     * Optionally the model is brought to steady state after the computation.
     */
    double getEE(const std::string& reactionName, const std::string& parameterName,
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
    double getScaledFloatingSpeciesElasticity(const std::string& reactionName,
            const std::string& speciesName);

    /**
     * Get a single species elasticity value
     * IMPORTANT:
     * Assumes that the reaction rates have been precomputed at the operating point !!
     */
    double getUnscaledSpeciesElasticity(int reactionId, int speciesIndex);


    /******************************* Steady State Section *************************/
    #if (1) /**********************************************************************/
    /******************************************************************************/

    /**
     * Compute the steady state of the model, returns the sum of squares of the
     * solution
     */
    double steadyState();

    /**
     * returns the current set of steady state selections.
     */
    std::vector<rr::SelectionRecord>& getSteadyStateSelections();

    /**
     * parses the given list of strings and generates selections records
     * which will be used for the steady state selections.
     */
    void setSteadyStateSelections(const std::vector<std::string>&
            steadyStateSelections);

    /**
     * makes a copy of an existing list of selection records. These will be
     * saved and used for selection values in getSteadyStateValues().
     */
    void setSteadyStateSelections(const std::vector<rr::SelectionRecord>&
            steadyStateSelections);

    /**
     * Performs a steady state calculation (evolves the system to a steady
     * state), then calculates and returns the set of values specifed by
     * the steady state selections.
     */
    std::vector<double> getSteadyStateValues();

    /******************************* End Steady State Section *********************/
    #endif /***********************************************************************/
    /******************************************************************************/



    /******** !!! DEPRECATED INTERNAL METHODS * THESE WILL BE REMOVED!!! **********/
    #if (1) /**********************************************************************/
    /******************************************************************************/

    #ifndef SWIG // deprecated methods not SWIG'ed

    /**
     * @internal
     * @deprecated
     */
    RR_DEPRECATED(int getNumberOfReactions());

    /**
     * @internal
     * @deprecated
     */
    RR_DEPRECATED(double getReactionRate(const int& index));

    /**
     * @deprecated
     * @internal
     * Returns the rate of changes of a species by its index
     */
    RR_DEPRECATED(double getRateOfChange(const int& index));

    /**
     * @internal
     * @deprecated
     */
    RR_DEPRECATED(std::vector<std::string> getRateOfChangeIds());

    /**
     * @internal
     * @deprecated
     */
    RR_DEPRECATED(std::vector<std::string> getConservedMoietyIds());

    /**
     * @internal
     * @deprecated
     */
    RR_DEPRECATED(std::vector<double> getConservedMoietyValues());

    /**
     * @internal
     * @deprecated
     */
    RR_DEPRECATED(int getNumberOfCompartments());

    /**
     * @internal
     * @deprecated
     */
    RR_DEPRECATED(void setCompartmentByIndex(const int& index, const double& value));

    /**
     * @internal
     * @deprecated
     */
    RR_DEPRECATED(double getCompartmentByIndex(const int& index));

    /**
     * @internal
     * @deprecated
     */
    RR_DEPRECATED(std::vector<std::string> getCompartmentIds());

    /**
     * @internal
     * @deprecated
     */
    RR_DEPRECATED(int getNumberOfBoundarySpecies());

    /**
     * @internal
     * @deprecated
     */
    RR_DEPRECATED(void setBoundarySpeciesByIndex(const int& index, const double& value));

    /**
     * @internal
     * @deprecated
     */
    RR_DEPRECATED(double getBoundarySpeciesByIndex(const int& index));

    /**
     * @internal
     * @deprecated
     */
    RR_DEPRECATED(std::vector<double> getBoundarySpeciesConcentrations());

    /**
     * @internal
     * @deprecated
     */
    RR_DEPRECATED(void setBoundarySpeciesConcentrations(const std::vector<double>& values));

    /**
     * @internal
     * @deprecated
     */
    RR_DEPRECATED(std::vector<std::string> getBoundarySpeciesIds());

    /**
     * @internal
     * @deprecated
     */
    RR_DEPRECATED(int getNumberOfFloatingSpecies());

    /**
     * @internal
     * @deprecated
     */
    RR_DEPRECATED(double getFloatingSpeciesByIndex(int index));

    /**
     * @internal
     * @deprecated
     */
    RR_DEPRECATED(void setFloatingSpeciesByIndex(int index, double value));

    /**
     * @internal
     * @deprecated
     */
    RR_DEPRECATED(std::vector<double> getFloatingSpeciesConcentrations());

    /**
     * @internal
     * @deprecated
     */
    RR_DEPRECATED(std::vector<double> getFloatingSpeciesInitialConcentrations());

    /**
     * @internal
     * @deprecated
     */
    RR_DEPRECATED(void setFloatingSpeciesConcentrations(const std::vector<double>& values));

    /**
     * @internal
     * @deprecated
     */
    RR_DEPRECATED(void setFloatingSpeciesInitialConcentrationByIndex(const int& index,
            const double& value));

    /**
     * @internal
     * @deprecated
     */
    RR_DEPRECATED(void setFloatingSpeciesInitialConcentrations(const std::vector<double>& values));

    /**
     * @internal
     * @deprecated
     */
    RR_DEPRECATED(std::vector<std::string> getFloatingSpeciesIds());

    /**
     * @internal
     * @deprecated
     */
    RR_DEPRECATED(std::vector<std::string> getFloatingSpeciesInitialConditionIds());

    /**
     * @internal
     * @deprecated use ExecutableModel::getNumGlobalParameters
     */
    RR_DEPRECATED(int getNumberOfGlobalParameters());

    /**
     * @internal
     * @deprecated use ExecutableModel::setGlobalParameterValues
     */
    RR_DEPRECATED(void setGlobalParameterByIndex(const int index, const double value));

    /**
     * @internal
     * @deprecated use ExecutableModel::getGlobalParameterValues
     */
    RR_DEPRECATED(double getGlobalParameterByIndex(const int& index));

    /**
     * @internal
     * @deprecated use ExecutableModel::getGlobalParameterValues
     */
    RR_DEPRECATED(std::vector<double> getGlobalParameterValues());

    /**
     * @internal
     * @deprecated use ExecutableModel::getGlobalParameterIds
     */
    RR_DEPRECATED(std::vector<std::string> getGlobalParameterIds());

    /**
     * The C back end requires this to be called to update
     * model variables if anyting is changes. Does nothing
     * in LLVM back end as everything is automatically handled
     * with lazy evaluation.
     *
     * @internal
     * @deprecated
     */
    void evalModel();

    /**
     * @internal
     * @deprecated
     */
    RR_DEPRECATED(int getNumberOfDependentSpecies());

    /**
     * @internal
     * @deprecated
     */
    RR_DEPRECATED(int getNumberOfIndependentSpecies());


    /**
     * @internal
     * @deprecated, use ExecutableModel::getReactionRates
     */
    RR_DEPRECATED(std::vector<double> getReactionRates());

    /**
     * @internal
     * @deprecated
     * returns a list of reaction ids obtained from
     * ExecutableModel::getReactionId
     */
    RR_DEPRECATED(std::vector<std::string> getReactionIds());

    /**
     * @internal
     * @deprecated
     *
     * C backend only
     * set the location where the ModelGenerator creates temporary files, such
     * as shared libraries.
     */
    void setTempDir(const std::string& folder);

    /**
     * @internal
     * @deprecated
     *
     * legacy C backend only
     * get the ModelGenerator's temporary file directory.
     */
    std::string getTempDir();

    #endif // #ifndef SWIG


    /******** !!! DEPRECATED INTERNAL METHODS * THESE WILL BE REMOVED!!! **********/
    #endif  /**********************************************************************/
    /******************************************************************************/

private:


    int createDefaultSteadyStateSelectionList();
    int createDefaultTimeCourseSelectionList();

    /**
     * copies the current selection values into the n'th row of the
     * given matrix
     */
    void getSelectedValues(ls::DoubleMatrix& results, int nRow,
            double currentTime);

    /**
     * copies the current selection values into the given vector.
     */
    void getSelectedValues(std::vector<double> &results, double currentTime);

    bool populateResult();


    double getNthSelectedOutput(unsigned index, double currentTime);

    enum VariableType
    {
        vtSpecies = 0, vtFlux
    };

    double getVariableValue(const VariableType variableType,
            const int variableIndex);

    /**
     * the LibStruct is normally null, only created on demand here.
     */
    LibStructural* getLibStruct();

    /**
     * If the specified integrator does not exist, create it, and point the
     * integrator pointer to it.
     */
    void updateIntegrator();

    bool createDefaultSelectionLists();

    /**
     * creates a selection list from the amounts / conc / variables ivars of the
     * SimulationOptions struct.
     */
    int createTimeCourseSelectionList();




    std::vector<SelectionRecord> getSelectionList();

    /**
     * sets the options and updates the integrator an any
     * other depedent bits.
     */
    void _setSimulateOptions(const SimulateOptions* opt);

    /**
     * private implementation class, can only access if inside
     * the implementation file.
     */
    class RoadRunnerImpl* impl;
};

}

#endif

/*! \mainpage RoadRunner C++ Library

 \par
 This document describes the application programming interface (API) of RoadRunner, an open source (BSD) library for computing structural characteristics of cellular networks.
 \par

 \par Dependencies
 The RoadRunner library depend on several third-party libraries, CLapack, libSBML (libxml2), Sundials, NLEQ, and Poco. These are provided with the binary installation where necessary.
 \par

 \author Andy Somogyi (andy.somogyi@gmail.com, somogyie@indiana.edu)
 \author Totte Karlsson (totte@dunescientific.com)
 \author Herbert M. Sauro  (hsauro@u.washington.edu)

 \par
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

 \li Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

 \li Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

 \li Neither the name of University of Washington nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

 \par

 */
