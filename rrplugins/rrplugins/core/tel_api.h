#pragma once

//Debashish Roy
// API used to connect roadrunner functionality with plugins

#include "../../../wrappers/C/rrc_types.h"		//decleration of types such as rrhandle		

#if defined(__cplusplus)
	namespace rrc { 
	extern "C" {
#endif

typedef struct
{
	const char* (*pluginName)();	 
} TPluginInterface;


typedef struct {			// THostInterface

	/*!
	 \brief Initialize a new roadRunner instance and return a handle to it.
	 \return Returns a RoadRunner instance, returns null if it fails
	 \ingroup initialization
	*/
	RRHandle  (*createRRInstance)(void);

	/*!
	 \brief Retrieve info about current state of roadrunner, e.g. loaded model, conservationAnalysis etc.
	 \param[in] handle Handle to a RoadRunner instance
	 \return Returns null if it fails, otherwise it returns a string with the info
	 \ingroup utility
	*/
	char*   (*getInfo)(RRHandle handle);

	/*!
	\brief Load a model from an SBML string
	\param[in] handle Handle to a RoadRunner instance
	\param[in] sbml string
	\return Returns true if successful
	\ingroup loadsave
	*/
	bool  (*loadSBML)(RRHandle handle, const char* sbml);

	/*!
	\brief Carry out a time-course simulation based on the given arguments, time start,
	time end and number of points.

	\param[in] handle Handle to a RoadRunner instance
	\param[in] timeStart Time start
	\param[in] timeEnd Time end
	\param[in] numberOfPoints Number of points to generate
	\return Returns an array (RRCDataPtr) of columns containing the results of the
	simulation including string labels for the individual columns. The client is
	responsible for freeing the resulting RRCDataPtr structure.
	\ingroup simulation
	*/

	RRCDataPtr (*simulateEx)(RRHandle handle, const double timeStart, const double timeEnd, const int numberOfPoints);

	/*!
	\brief Specify the current steady state solver to be used for simulation.
	\description This method instantiates a new steady state solver of the given type (e.g. cvode, gillespie) if
	one does not currently exist. Otherwise, the existing steady state solver of this type is used.
	\param[in] handle Handle to a RoadRunner instance.
	\param[in] nameOfSteadyStateSolver Name of the steady state solver to be used.
	\return Returns True if successful.
	\ingroup simopts
	*/

	int (*setCurrentSteadyStateSolver)(RRHandle handle, char* nameOfSteadyStateSolver);

	/*!
	\brief Set the boolean value for a specific steady state solver setting.
	\param[in] handle Handle to a RoadRunner instance.
	\param[in] paramterName Name of the steady state solver setting.
	\param[in] value The boolean value for the steady state solver setting.
	\return Returns True if successful.
	\ingroup simopts
	*/
	
	int (*setCurrentSteadyStateSolverParameterBoolean)(RRHandle handle, char* parameterName, int value);

	/*!
	\brief Set the double value for a specific steady state solver setting.
	\param[in] handle Handle to a RoadRunner instance.
	\param[in] paramterName Name of the steady state solver setting.
	\param[in] value The double value for the steady state solver setting.
	\return Returns True if successful.
	\ingroup simopts
	*/
	int (*setCurrentSteadyStateSolverParameterDouble)(RRHandle handle, char* parameterName, double value);


	/*!
	\brief Set the path to the temporary folder where the C code will be stored
	When RoadRunner is run in C generation mode its uses a temporary folder to store the
	generated C source code. This method can be used to set the temporary folder path if necessary.
	\param[in] handle Handle to a RoadRunner instance
	\param[in] folder Pointer to string holding folder path
	\return Returns true if successful
	\ingroup utility                                                 onsole
	*/
	bool (*setTempFolder)(RRHandle handle, const char* folder);

	/*!
	\brief Compute the steady state of the current model
	Example: \code status = steadyState (rrHandle, &closenessToSteadyState); \endcode
	\param[in] handle Handle to a RoadRunner instance
	\param value This value is set during the call and indicates how close the solution is to the steady state.
	The smaller the value the better. Values less than 1E-6 usually indicate a steady state has been found. If necessary
	call the method a second time to improve the solution.
	\return Returns true if successful
	\ingroup steadystate
	*/
	bool (*steadyState)(RRHandle handle, double* value);

	/*!
	\brief Set the concentration for a particular boundary species.
	\param[in] handle Handle to a RoadRunner instance
	\param index The index to the boundary species (corresponds to position in getBoundarySpeciesIds(void))
	\param value The concentration of the species to set
	\return Returns true if successful
	\ingroup boundary
	*/
	bool (*setBoundarySpeciesByIndex)(RRHandle handle, const unsigned int index, const double value);

	/*!
	\brief Set the value for a given symbol, use getAvailableTimeCourseSymbols(void) for a list of symbols
	Example: \code status = setValue (rrHandle, "S1", 0.5); \endcode
	\param[in] handle Handle to a RoadRunner instance
	\param symbolId The symbol that we wish to set the value
	\param value The value that will be set to the symbol
	\return Returns true if successful
	\ingroup state
	*/
	bool (*setValue)(RRHandle handle, const char* symbolId, const double value);

	/*!
	\brief Carry out a time-course simulation. setTimeStart, setTimeEnd,
	setNumPoints, etc are used to set the simulation characteristics.
	\param[in] handle Handle to a RoadRunner instance
	\return Returns an array (RRCDataPtr) of columns containing the results of the
	simulation including string labels for the individual columns. The client is
	responsible for freeing the resulting RRCDataPtr structure.
	\ingroup simulation
	*/
	RRCDataPtr (*simulate)(RRHandle handle);

	/*!
	\brief Retrieve the concentration for a particular floating species.
	\param[in] handle Handle to a RoadRunner instance
	\param index The index to the boundary species (corresponds to position in getBoundarySpeciesIds(RRHandle handle))
	\param value The value returned by the method
	\return Returns true if successful
	\ingroup boundary
	*/
	bool (*getBoundarySpeciesByIndex)(RRHandle handle, const int index, double* value);// I have modified this and C API

	/*!
	\brief Retrieve the global parameter value
	\param[in] handle Handle to a RoadRunner instance
	\param index The index to the global parameter (corresponds to position in getGlobalParametersIds(RRHandle handle))
	\param value The value returned by the method
	\return Returns true if successful, false otherwise
	\ingroup parameters
	*/
	bool (*getGlobalParameterByIndex)(RRHandle handle, const int index, double* value);

	/*!
	\brief Get the selection list for the steady state analysis
	\param[in] handle Handle to a RoadRunner instance
	\return Returns null if it fails, otherwise it returns a list of strings representing symbols in the selection list
	\ingroup steadystate
	*/
	RRStringArrayPtr (*getSteadyStateSelectionList)(RRHandle handle);

	/*!
	\brief Obtain the list of global parameter Ids
	\param[in] handle Handle to a RoadRunner instance
	\return Returns null if it fails, if successful it returns a pointer to a RRStringArrayPtr struct
	\ingroup parameters
	*/
	RRStringArrayPtr (*getGlobalParameterIds)(RRHandle handle);

	/*!
	\brief Obtain the list of boundary species Ids
	\param[in] handle Handle to a RoadRunner instance
	\return Returns null if it fails, if successful it returns a pointer to a RRStringArrayPtr struct
	\ingroup boundary
	*/
	RRStringArrayPtr (*getBoundarySpeciesIds)(RRHandle handle);

	// Flags/Options
	/*!
	\brief Enable or disable conservation analysis
	\param[in] handle Handle to a RoadRunner instance
	\param[in] On_Or_Off Set true to switch on conservation analysis
	\return Returns true if successful
	\ingroup initialization
	*/
	bool (*setComputeAndAssignConservationLaws)(RRHandle handle, const bool On_Or_Off);


	/*!
    \return the number of floating species pointed by the model by handle
	*/

	int (*_getNumIndFloatingSpecies)(RRHandle handle);

	/*!
    \return the number of rate rules pointed by the model by handle
	*/
 	int (*_getNumRateRules)(RRHandle handle);


 	/*!
 	 \brief Retrieve in a vector the concentrations for all the floating species

 	 Example: \code RRVectorPtr values = getFloatingSpeciesConcentrations (void); \endcode

 	 \param[in] handle Handle to a RoadRunner instance
 	 \return Returns the vector of floating species concentrations or null if an error occurred
 	 \ingroup floating
 	*/
 	RRVectorPtr (*getFloatingSpeciesConcentrations)(RRHandle handle);

	bool (*setFloatingSpeciesConcentrations)(RRHandle handle, const RRVectorPtr vec);

	RRVectorPtr (*getRatesOfChange)(RRHandle handle);

	/*!
	\return the time for getsteadyvectorrate
	*/
	double (*_getTime)(RRHandle handle);

	/*!
		\return getsteadyvectorrate
	*/
	int (*_getStateVector)(RRHandle handle);

	/*!
		\return getsteadyvectorrate
	*/
	void (*_getStateVectorRate)(RRHandle handle, double time, double* value);

	/*Print the current version of roadrunner*/

	char* (*getVersionStr)();

	// --------------------------------------------------------------------------------
	// Reset methods
	// --------------------------------------------------------------------------------

	/*!
	\brief Resets all variables of the model to their current initial values.
	Does not change the parameters.
	\param[in] handle Handle to a RoadRunner instance
	\return Boolean indicating success
	\ingroup reset
	*/
	bool (*reset)(RRHandle handle);

	/*!
	\brief Retrieve the result of the last simulation.
	\param[in] handle Handle to a RoadRunner instance
	\return Returns an array (RRCDataPtr) of columns containing the results of the
	simulation including string labels for the individual columns. The client is
	responsible for freeing the resulting RRCDataPtr structure.
	\ingroup simulation
	*/
	RRCDataPtr (*getSimulationResult)(RRHandle handle);

	/*!
	\brief Set the selection list for output from simulate(void) or simulateEx(void)
	Use getAvailableTimeCourseSymbols(void) to retrieve the list of all possible symbols.
	Example: \code setTimeCourseSelectionList ("Time, S1, J1, J2"); \endcode
	or
	setTimeCourseSelectionList ("Time S1 J1 J2")
	\param[in] handle Handle to a RoadRunner instance
	\param[in] list A string of Ids separated by spaces <b>or</b> comma characters
	\return Returns true if successful
	\ingroup simulation
	*/
	bool (*setTimeCourseSelectionList)(RRHandle handle, const char* list);

} THostInterface;

//THostInterface hostInterfaceMain;

#if defined( __cplusplus)
		}//extern
	}//namespace
#endif

