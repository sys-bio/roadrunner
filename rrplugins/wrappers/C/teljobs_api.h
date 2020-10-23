#ifndef teljobs_apiH
#define teljobs_apiH
#include "telplugins_exporter.h"
#include "telplugins_types.h"
//---------------------------------------------------------------------------

#if defined(__cplusplus)
namespace tlpc
{

extern "C" {
#endif

/*!
 \brief Typedef for callback function, taking a void* parameter
 \ingroup multithreading
*/
typedef void (tlp_cc *callBackFunc)(void*);

/*!
 \brief Run a roadrunner simulation in a thread
 \param[in] handle Handle to a RoadRunner instance
 \return Returns a handle to the Job if succesful, otherwise returns NULL
 \ingroup multiThreading
*/
TLP_C_DS TELHandle tlp_cc simulateJob(TELHandle handle);

/*!
 \brief Run a roadrunner simulation in a thread
 \param[in] handle Handle to a RoadRunner instance
 \param timeStart,timeEnd,numberOfPoints Parameters for the simulation
 \param fn1,fn2 Callback functions that are used internally in the thread.
 \param userData User supplied data that can be used in the supplied callback functions
 \return Returns a handle to the Job if succesful, otherwise returns NULL
 \ingroup multiThreading
*/
TLP_C_DS TELHandle tlp_cc simulateJobEx(TELHandle handle, double timeStart,
                                                        double timeEnd,    int numberOfPoints,
                                                        callBackFunc fn1, callBackFunc fn2, void* userData);

/*!
 \brief Run multiple simulations in a set of threads
 \param[in] handles Handle to a list of RoadRunner instances
 \param[in] nrOfThreads Number of threads to allocate in order to execute the jobs
 \return Returns a handle to a list of jobs if succesful, otherwise returns NULL
 \ingroup multiThreading
*/
TLP_C_DS TELHandle tlp_cc simulateJobs(TELHandle handles, int nrOfThreads);


/*!
 \brief Run multiple simulations in a set of threads
 \param[in] handles Handle to a list of RoadRunner instances
 \param[in] nrOfThreads Number of threads to allocate in order to execute the jobs
 \param timeStart,timeEnd,numberOfPoints Parameters for the simulation
 \param fn1,fn2 Callback functions that are used internally in the thread.
 \param userData User supplied data that can be used in the supplied callback functions
 \return Returns a handle to a list of jobs if succesful, otherwise returns NULL
 \ingroup multiThreading
*/TLP_C_DS TELHandle tlp_cc simulateJobsEx(TELHandle handles, int nrOfThreads,
                                                        double timeStart, double timeEnd, int numberOfPoints,
                                                        callBackFunc fn1, callBackFunc fn2, void* userData);

/*!
 \brief Load a model from a SBML file into a RoadRunner instances, using a Job
 \param[in] handle Handle to a RoadRunner instance
 \param[in] fileName file name (with optional full path) to a file that holds the SBML model
 \return Returns a handle to the Job if succesful, otherwise returns NULL
 \ingroup multiThreading
*/

TLP_C_DS TELHandle tlp_cc loadSBMLFromFileJob(TELHandle handle, const char* fileName);

/*!
 \brief Load a model from a SBML file into a RoadRunner instances, using a Job
 \param[in] handle Handle to a RoadRunner instance
 \param[in] fileName file name (or full path) to file that holds the SBML model
 \param[in] reCompile Boolean that forces recompilation if true. If false, no compilation occur if model dll exists
 \return Returns a handle to the Job if succesful, otherwise returns NULL
 \ingroup multiThreading
*/
TLP_C_DS TELHandle tlp_cc loadSBMLFromFileJobEx(TELHandle handle, const char* fileName, bool reCompile);

/*!
 \brief Load a model from a SBML file into a set of RoadRunner instances
 \param[in] handles Handle to a list of RoadRunner instances
 \param[in] fileName file name (or full path) to file that holds the SBML model
 \param[in] nrOfThreads Number of threads to allocate in order to execute the job
 \return Returns a handle to the Jobs if succesful, otherwise returns NULL
 \ingroup multiThreading
*/
TLP_C_DS TELHandle tlp_cc loadSBMLFromFileJobs(TELHandle handles, const char* fileName, int nrOfThreads);

/*!
 \brief Load a model from a SBML text string into a RoadRunner instances, using a Job
 \param[in] handle Handle to a RoadRunner instance
 \param[in] sbml String that holds the SBML model
 \return Returns a handle to the Job if succesful, otherwise returns NULL
 \ingroup multiThreading
*/
TLP_C_DS TELHandle tlp_cc loadSBMLJob(TELHandle handle, const char* sbml);

/*!
 \brief Load a model from a SBML text string into a RoadRunner instances, using a Job
 \param[in] handle Handle to a RoadRunner instance
 \param[in] sbml String that holds the SBML model
 \param[in] reCompile Boolean that forces recompilation if true. If false, no compilation occur if model dll exists
 \return Returns a handle to the Job if succesful, otherwise returns NULL
 \ingroup multiThreading
*/
TLP_C_DS TELHandle tlp_cc loadSBMLJobEx(TELHandle handle, const char* sbml, bool reCompile);

/*!
 \brief Load a model from a SBML tesxt string into a set of RoadRunner instances
 \param[in] handles Handle to a list of RoadRunner instances
 \param[in] sbml String that holds the SBML model
 \param[in] nrOfThreads Number of threads to allocate in order to execute the job
 \return Returns a handle to the Jobs if succesful, otherwise returns NULL
 \ingroup multiThreading
*/
TLP_C_DS TELHandle tlp_cc loadSBMLJobs(TELHandle handles, const char* sbml, int nrOfThreads);

/*!
 \brief Load a model from a SBML file into a set of RoadRunner instances
 \param[in] handles Handle to a list of RoadRunner instances
 \param[in] sbml string that holds the SBML model
 \param[in] nrOfThreads Number of threads to allocate in order to execute the job
 \param[in] force_recompile Indicates if a forced recompilation of the model should be executed
 \return Returns a handle to the Jobs if succesful, otherwise returns NULL
 \ingroup multiThreading
*/
TLP_C_DS TELHandle tlp_cc loadSBMLJobsEx(TELHandle handles, const char* sbml, int nrOfThreads, bool force_recompile);

/*!
 \brief Wait for jobs in thread to finish
 \param[in] handle Handle to a RoadRunner job
 \return Returns true if thread finsihed up properly, otherwise returns false
 \ingroup multiThreading
*/
TLP_C_DS bool tlp_cc waitForJob(TELHandle handle);

/*!
 \brief Wait for jobs in thread pool to finish
 \param[in] handle Handle to a list of RoadRunner jobs
 \return Returns true if threadpool finished up properly, otherwise returns false
 \ingroup multiThreading
*/
TLP_C_DS bool tlp_cc waitForJobs(TELHandle handle);

/*!
 \brief Check if there are work being done on a job
 \param[in] handle Handle to a RoadRunner job
 \return Returns true if there are running threads, otherwise returns false
 \ingroup multiThreading
*/
TLP_C_DS bool tlp_cc isJobFinished(TELHandle handle);

/*!
 \brief Check if there are work being done on jobs
 \param[in] handle Handle to a list of RoadRunner jobs
 \return Returns true if there are running threads, otherwise returns false
 \ingroup multiThreading
*/
TLP_C_DS bool tlp_cc areJobsFinished(TELHandle handle);

/*!
 \brief Get number of remaining jobs in a job list
 \param[in] handle Handle to a list of RoadRunner jobs
 \return Returns number of remaining, unfinished jobs. Returns -1 on failure
 \ingroup multiThreading
*/
TLP_C_DS int tlp_cc getNumberOfRemainingJobs(TELHandle handle);

/*!
 \brief Free a job handle
 \param[in] handle Handle to a roadrunner job
 \return Returns true or false indicating if the function was succesful
 \ingroup multiThreading
*/
TLP_C_DS bool tlp_cc freeJob(TELHandle handle, RRJobType jt);

/*!
 \brief Free a jobs handle
 \param[in] handle Handle to a list of roadrunner jobs
 \return Returns true or false indicating if the function was succesful
 \ingroup multiThreading
*/
TLP_C_DS bool tlp_cc freeJobs(TELHandle handle);

#if defined(__cplusplus)
}}    //Extern "C" //namespace
#endif

#endif
