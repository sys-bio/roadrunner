#pragma hdrstop
#include "CVODEIntegrator.h"
#include "rrExecutableModel.h"
#include "rrException.h"
#include "rrLogger.h"
#include "rrStringUtils.h"
#include "rrException.h"
#include "rrConfig.h"
#include "rrUtils.h"

#include <cvode/cvode.h>
#include <cvode/cvode_dense.h>
#include <nvector/nvector_serial.h>
#include <cstring>
#include <iomanip>
#include <math.h>
#include <map>
#include <algorithm>
#include <limits>
#include <assert.h>
#include <Poco/Logger.h>


using namespace std;
namespace rr
{
	/* Store the name, description, and hint as global variables to ensure consistency between
	* between IntegratorRegistrar and internal integrator methods (ie. getIntegratorName()). */
	const char* gCVODEName = "cvode";
	const char* gCVODEDesc = "CVODE description.";
	const char* gCVODEHint = "CVODE hint.";

	// Creates a new CVODE integrator object
	static Integrator* makeCVODEIntegrator(ExecutableModel *m)
	{
		return new CVODEIntegrator(m);
	}

	// Register with factory
	static int cvode_result = IntegratorFactory::getInstance().registerIntegrator(IntegratorRegistrar(gCVODEName, gCVODEDesc, gCVODEHint, makeCVODEIntegrator));

	const int CVODEIntegrator::mDefaultMaxNumSteps = 10000;
	const int CVODEIntegrator::mDefaultMaxAdamsOrder = 12;
	const int CVODEIntegrator::mDefaultMaxBDFOrder = 5;

	int cvodeDyDtFcn(realtype t, N_Vector cv_y, N_Vector cv_ydot, void *userData);
	int cvodeRootFcn(realtype t, N_Vector y, realtype *gout, void *userData);

	// Sets the value of an element in a N_Vector object
	inline void SetVector(N_Vector v, int Index, double Value)
	{
		double *data = NV_DATA_S(v);
		data[Index] = Value;
	}

	inline double GetVector(N_Vector v, int Index)
	{
		double *data = NV_DATA_S(v);
		return data[Index];
	}

	/**
	* Purpose
	* This function processes error and warning messages from CVODE and its
	* sub-modules.
	*
	* Arguments:
	* error_code: is the error code.
	* module: is the name of the CVODE module reporting the error.
	* function: is the name of the function in which the error occurred.
	* msg: is the error message.
	* eh_data: is a pointer to user data, the same as the eh_data parameter
	* passed to CVodeSetErrHandlerFn.
	* Return value: A CVErrHandlerFn function has no return value.
	*
	* Notes:
	* error_code is negative for errors and positive (CV_WARNING) for warnings.
	* If a function that returns a pointer to memory encounters an error,
	* it sets error_code to 0.
	*/
	static void cvodeErrHandler(int error_code, const char *module, const char *function, char *msg, void *eh_data);

	/**
	* decode the cvode error code to a string
	*/
	static std::string cvodeDecodeError(int cvodeError, bool exInfo = true);

	/**
	* macro to throw a (hopefully) usefull error message
	*/
#define handleCVODEError(errCode) \
	{ std::string _err_what = std::string("CVODE Error: ") + \
	cvodeDecodeError(errCode); \
	throw IntegratorException(_err_what, std::string(__FUNC__)); }

	void CVODEIntegrator::setListener(IntegratorListenerPtr p)
	{
		listener = p;
	}

	IntegratorListenerPtr CVODEIntegrator::getListener()
	{
		return listener;
	}

	CVODEIntegrator::CVODEIntegrator(ExecutableModel *aModel)
		:
		mStateVector(NULL),
		mCVODE_Memory(NULL),
		lastEventTime(0),
		mModel(aModel),
		stateVectorVariables(false),
		variableStepPendingEvent(false),
		variableStepTimeEndEvent(false),
		variableStepPostEventState(0)
	{
		Log(Logger::LOG_INFORMATION) << "creating CVODEIntegrator";

		// Set default integrator settings.
		addSetting("stiff", false, "stiff hint.", "stiff description.");
		addSetting("variable_step_size", false, "Perform a variable time step simulation.", "Enabling this setting will allow the integrator to adapt the size of each time step. This will result in a non-uniform time column.");
		addSetting("multiple_steps", false, "multiple steps hint.", "multiple steps description.");
		addSetting("initial_time_step", 0.0, "initial time step hint.", "initial time step description.");
		addSetting("minimum_time_step", 0.0, "minimum time step hint.", "minimum time step description.");
		addSetting("maximum_time_step", 0.0, "maximum time step hint.", "maximum time step description.");
		addSetting("maximum_num_steps", mDefaultMaxNumSteps, "Maximum number of steps hint.", "Maximum number of steps description.");
		addSetting("maximum_adams_order", mDefaultMaxAdamsOrder, "Maximum Adams Order hint. (int)", "Maximum Adams Order description.");
		addSetting("maximum_bdf_order", mDefaultMaxBDFOrder, "Maximum BDF Order hint. (int)", "Maximum BDF Order description.");
		addSetting("relative_tolerance", 1e-6, "Relative tolerance hint.", "Relative tolerance description.");
		addSetting("absolute_tolerance", 1e-12, "Absolute tolerance hint.", "Absolute tolerance description.");
		CVODEIntegrator::loadConfigSettings();

		if (aModel)
		{
			createCVode();

			// allocate space for the event status array
			eventStatus = std::vector<unsigned char>(mModel->getEventTriggers(0, 0, 0), false);
		}

		// Update parameter settings within CVODE.
		updateCVODE();
	}

	CVODEIntegrator::~CVODEIntegrator()
	{
		delete[] variableStepPostEventState;
		freeCVode();
	}

	void CVODEIntegrator::loadConfigSettings()
	{
		Integrator::loadConfigSettings();
		// Load settings specific to CVODE integrator

		CVODEIntegrator::setValue("absolute_tolerance", Config::getDouble(Config::CVODE_MIN_ABSOLUTE));
		CVODEIntegrator::setValue("relative_tolerance", Config::getDouble(Config::CVODE_MIN_RELATIVE));
	}

	void CVODEIntegrator::loadSBMLSettings(std::string const& filename)
	{
		if (!filename.size())
		{
			Log(Logger::LOG_ERROR) << "Empty file name for settings file";
		}
		else
		{
			map<string, string> options;
			map<string, string>::iterator it;
			//Read each line in the settings file
			vector<string> lines = getLinesInFile(filename);
			for (int i = 0; i < lines.size(); i++)
			{
				vector<string> line = splitString(lines[i], ":");
				if (line.size() == 2)
				{
					options.insert(pair<string, string>(line[0], line[1]));
				}
				else
				{
					Log(lDebug2) << "Empty line in settings file: " << lines[i];
				}
			}

			Log(lDebug3) << "Settings File =============";
			for (it = options.begin(); it != options.end(); it++)
			{
				Log(lDebug) << (*it).first << " => " << (*it).second;
			}
			Log(lDebug) << "===========================";

			//Assign values
			it = options.find("absolute");
			if (it != options.end())
			{
				CVODEIntegrator::setValue("absolute_tolerance", std::abs(toDouble((*it).second)));
			}

			it = options.find("relative");
			if (it != options.end())
			{
				CVODEIntegrator::setValue("relative_tolerance", std::abs(toDouble((*it).second)));
			}
		}
	}

	std::string CVODEIntegrator::getIntegratorName() const
	{
		return gCVODEName;
	}

	std::string CVODEIntegrator::getIntegratorDescription() const
	{
		return gCVODEDesc;
	}

	std::string CVODEIntegrator::getIntegratorHint() const
	{
		return gCVODEHint;
	}

	Integrator::IntegrationMethod CVODEIntegrator::getIntegrationMethod() const
	{
		return Integrator::IntegrationMethod::Deterministic;
	}

	void CVODEIntegrator::setValue(string key, const Variant& val)
	{
		Integrator::setValue(key, val);

		/*	In addition to typically value-setting behavior, some settings require further changes 
			within CVODE. */
		if (key == "maximum_bdf_order")
		{
			CVodeSetMaxOrd(mCVODE_Memory, getValueAsInt("maximum_bdf_order"));
		}
		else if (key == "maximum_adams_order")
		{
			CVodeSetMaxOrd(mCVODE_Memory, getValueAsInt("maximum_adams_order"));
		}
		else if (key == "initial_time_step")
		{
			CVodeSetInitStep(mCVODE_Memory, getValueAsDouble("initial_time_step"));
		}
		else if (key == "minimum_time_step")
		{
			CVodeSetMinStep(mCVODE_Memory, getValueAsDouble("minimum_time_step"));
		}
		else if (key == "maximum_time_step")
		{
			CVodeSetMaxStep(mCVODE_Memory, getValueAsDouble("maximum_time_step"));
		}
		else if (key == "maximum_num_steps")
		{
			CVodeSetMaxNumSteps(mCVODE_Memory, getValueAsInt("maximum_num_steps"));
		}
		else if (key == "stiff")
		{
			// If the integrator is changed from stiff to standard, we must re-create CVode.
			Log(Logger::LOG_INFORMATION) << "Integrator stiffness has been changed. Re-creating CVode.";
			freeCVode();
			createCVode();
		}
	}

	void CVODEIntegrator::updateCVODE()
	{
		if (mCVODE_Memory == 0)
		{
			return;
		}

		std::unordered_map<string, Variant>::const_iterator option;

		CVodeSetInitStep(mCVODE_Memory, getValueAsDouble("initial_time_step"));
		CVodeSetMinStep(mCVODE_Memory, getValueAsDouble("minimum_time_step"));
		CVodeSetMaxStep(mCVODE_Memory, getValueAsDouble("maximum_time_step"));
		CVodeSetMaxNumSteps(mCVODE_Memory, getValueAsInt("maximum_num_steps") > 0 ? getValueAsInt("maximum_num_steps") : mDefaultMaxNumSteps);
		setCVODETolerances();
	}

	void CVODEIntegrator::reInit(double t0)
	{
		// if we have no state vector variables and no events, we never
		// created an integrator.
		if (mCVODE_Memory == 0)
		{
			return;
		}

		int result = CVodeReInit(mCVODE_Memory, t0, mStateVector);

		if (result != CV_SUCCESS)
		{
			handleCVODEError(result);
		}

		setCVODETolerances();
	}

	double CVODEIntegrator::integrate(double timeStart, double hstep)
	{
		static const double epsilon = std::numeric_limits<double>::epsilon();

		Log(Logger::LOG_DEBUG) << "CVODEIntegrator::integrate(" << timeStart << ", " << hstep << ")";

		if (variableStepPendingEvent || variableStepTimeEndEvent)
		{
			return applyVariableStepPendingEvents();
		}

		double timeEnd = 0.0;
		double tout = timeStart + hstep;
		int strikes = 3;

		// Set itask based on step size settings.
		int itask = CV_ONE_STEP;

		if (getValueAsBool("multiple_steps"))
		{
			itask = CV_NORMAL;
		}

		if (getValueAsBool("variable_step_size"))
		{
			itask = CV_NORMAL;
		}


		// TODO: CONFIRM THIS WITH HERBERT, BUT I THINK THAT THIS WAS SET UP INCORRECTLY.
		/*
		const int itask = ((options.integratorFlags & MULTI_STEP)
		|| (options.integratorFlags & VARIABLE_STEP))
		? CV_ONE_STEP : CV_NORMAL;*/

		// loop until machine epislon
		while (tout - timeEnd >= epsilon)
		{
			// here we bail in case we have no ODEs set up with CVODE ... though we should
			// still at least evaluate the model function
			if (!haveVariables() && mModel->getNumEvents() == 0)
			{
				mModel->getStateVectorRate(tout, 0, 0);
				return tout;
			}

			double nextTargetEndTime = tout;

			if (mModel->getPendingEventSize() > 0 && mModel->getNextPendingEventTime(false) < nextTargetEndTime)
			{
				nextTargetEndTime = mModel->getNextPendingEventTime(true);
			}

			// event status before time step
			mModel->getEventTriggers(eventStatus.size(), 0, eventStatus.size() == 0 ? NULL : &eventStatus[0]);

			// time step
			int nResult = CVode(mCVODE_Memory, nextTargetEndTime, mStateVector, &timeEnd, itask);

			if (nResult == CV_ROOT_RETURN)
			{
				Log(Logger::LOG_DEBUG) << "Event detected at time " << timeEnd;

				bool tooCloseToStart = fabs(timeEnd - lastEventTime) > getValueAsBool("relative_tolerance");

				if (tooCloseToStart)
				{
					strikes = 3;
				}
				else
				{
					strikes--;
				}

				// the condition that we are to evaluate and apply events
				if (tooCloseToStart || strikes > 0)
				{
					lastEventTime = timeEnd;

					if (getValueAsBool("variable_step_size") && (timeEnd - timeStart > 2. * epsilon))
					{
						variableStepPendingEvent = true;
						assignResultsToModel();
						mModel->setTime(timeEnd - epsilon);
						if (listener)
						{
							listener->onTimeStep(this, mModel, timeEnd);
						}
						return timeEnd - epsilon;
					}

					// apply events, copy post event status into integrator state vector.
					applyEvents(timeEnd, eventStatus);

					if (listener)
					{
						listener->onEvent(this, mModel, timeEnd);
					}
				}
			}
			else if (nResult == CV_SUCCESS)
			{
				// copy integrator state vector into model
				assignResultsToModel();


				// need to check if an event occured at the exact time step,
				// if so, add an extra point if we're doing variable step
				if (getValueAsBool("variable_step_size") && (timeEnd - timeStart > 2. * epsilon))
				{
					// event status before time step
					mModel->getEventTriggers(eventStatus.size(), 0, &eventStatus[0]);
					// apply events and write state to variableStepPostEventState
					// model state is updated by events.
					int handled = mModel->applyEvents(timeEnd, &eventStatus[0],
						NULL, variableStepPostEventState);
					if (handled > 0) {
						// write original state back to model
						mModel->setTime(timeEnd - epsilon);
						assignResultsToModel();
						variableStepTimeEndEvent = true;
						lastEventTime = timeEnd;
						return timeEnd - epsilon;
					}
				}
				else {


					mModel->setTime(timeEnd);

					// only needs to be called after a reg time step completes, the applyEvents
					// called when a event root is found clears out all pending events and applies
					// them.
					applyPendingEvents(timeEnd);
				}

				if (listener)
				{
					listener->onTimeStep(this, mModel, timeEnd);
				}
			}
			else
			{
				handleCVODEError(nResult);
			}

			try
			{
				mModel->testConstraints();
			}
			catch (const std::exception& e)
			{
				Log(Logger::LOG_WARNING) << "Constraint Violated at time = " << timeEnd << ": " << e.what();
			}


			if (tout - timeEnd > epsilon)
			{
				timeStart = timeEnd;
			}
			Log(Logger::LOG_TRACE) << "time step, tout: " << tout << ", timeEnd: " << timeEnd;

			//if (options.integratorFlags & VARIABLE_STEP)
			if ((bool)getValue("variable_step_size"))
			{
				return timeEnd;
			}
		}
		return timeEnd;
	}

	void CVODEIntegrator::tweakTolerances()
	{
		double minAbs = Config::getDouble(Config::CVODE_MIN_ABSOLUTE);
		double minRel = Config::getDouble(Config::CVODE_MIN_RELATIVE);

		CVODEIntegrator::setValue("absolute_tolerance", std::min(CVODEIntegrator::getValueAsDouble("absolute_tolerance"), minAbs));
		CVODEIntegrator::setValue("relative_tolerance", std::min(CVODEIntegrator::getValueAsDouble("relative_tolerance"), minRel));

		Log(Logger::LOG_INFORMATION) << "tweaking CVODE tolerances to abs=" << CVODEIntegrator::getValueAsDouble("absolute_tolerance") << ", rel=" << CVODEIntegrator::getValueAsDouble("relative_tolerance");
	}

	bool CVODEIntegrator::haveVariables()
	{
		return stateVectorVariables;
	}

	void CVODEIntegrator::createCVode()
	{
		if (!mModel)
		{
			return;
		}

		assert(mStateVector == 0 && mCVODE_Memory == 0 &&
			"calling cvodeCreate, but cvode objects already exist");

		// still need cvode state vector size if we have no vars, but have
		// events, needed so root finder works.
		int allocStateVectorSize = 0;
		int realStateVectorSize = mModel->getStateVector(0);

		// cvode return code
		int err;

		if (realStateVectorSize > 0)
		{
			stateVectorVariables = true;
			allocStateVectorSize = realStateVectorSize;
		}
		else if (mModel->getNumEvents() > 0)
		{
			allocStateVectorSize = 1;
			stateVectorVariables = false;
		}
		else
		{
			stateVectorVariables = false;
			return;
		}

		// allocate and init the cvode arrays
		mStateVector = N_VNew_Serial(allocStateVectorSize);
		variableStepPostEventState = new double[allocStateVectorSize];

		for (int i = 0; i < allocStateVectorSize; i++)
		{
			SetVector(mStateVector, i, 0.);
		}

		if (getValueAsBool("stiff"))
		{
			Log(Logger::LOG_INFORMATION) << "using stiff integrator";
			mCVODE_Memory = (void*)CVodeCreate(CV_BDF, CV_NEWTON);
		}
		else
		{
			Log(Logger::LOG_INFORMATION) << "using non-stiff integrator";
			mCVODE_Memory = (void*)CVodeCreate(CV_ADAMS, CV_FUNCTIONAL);
		}

		assert(mCVODE_Memory && "could not create Cvode, CVodeCreate failed");

		if ((err = CVodeSetErrHandlerFn(mCVODE_Memory, cvodeErrHandler, NULL)) != CV_SUCCESS)
		{
			handleCVODEError(err);
		}

		// use non default CVODE value here, default is too short
		// for some sbml tests.
		CVodeSetMaxNumSteps(mCVODE_Memory, mDefaultMaxNumSteps);

		double t0 = 0.0;

		if ((err = CVodeSetUserData(mCVODE_Memory, (void*) this)) != CV_SUCCESS)
		{
			handleCVODEError(err);
		}

		if ((err = CVodeInit(mCVODE_Memory, cvodeDyDtFcn, t0, mStateVector)) != CV_SUCCESS)
		{
			handleCVODEError(err);
		}

		if (mModel->getNumEvents() > 0)
		{
			if ((err = CVodeRootInit(mCVODE_Memory, mModel->getNumEvents(),
				cvodeRootFcn)) != CV_SUCCESS)
			{
				handleCVODEError(err);
			}
			Log(Logger::LOG_TRACE) << "CVRootInit executed.....";
		}

		// only allocate this if we are using stiff solver.
		// otherwise, CVode will NOT free it if using standard solver.
		if (getValueAsBool("stiff"))
		{
			if ((err = CVDense(mCVODE_Memory, allocStateVectorSize)) != CV_SUCCESS)
			{
				handleCVODEError(err);
			}
		}

		setCVODETolerances();
		mModel->resetEvents();
	}

	void CVODEIntegrator::testRootsAtInitialTime()
	{
		vector<unsigned char> initialEventStatus(mModel->getEventTriggers(0, 0, 0), false);
		mModel->getEventTriggers(initialEventStatus.size(), 0, initialEventStatus.size() == 0 ? NULL : &initialEventStatus[0]);
		applyEvents(0, initialEventStatus);
	}

	void CVODEIntegrator::applyEvents(double timeEnd, vector<unsigned char> &previousEventStatus)
	{
		double *stateVector = mStateVector ? NV_DATA_S(mStateVector) : 0;
		mModel->applyEvents(timeEnd, previousEventStatus.size() == 0 ? NULL : &previousEventStatus[0], stateVector, stateVector);

		if (timeEnd > 0.0) {
			mModel->setTime(timeEnd);

			// copy state vector into cvode memory
			if (mStateVector)
			{
				mModel->getStateVector(NV_DATA_S(mStateVector));
			}

			// set tolerances and so forth.
			reInit(timeEnd);
		}
	}


	void CVODEIntegrator::applyPendingEvents(double timeEnd)
	{
		if (mModel) {
			// get current event triggered state
			mModel->getEventTriggers(eventStatus.size(), 0, eventStatus.size() == 0 ? NULL : &eventStatus[0]);
			int handled = mModel->applyEvents(timeEnd, eventStatus.size() == 0 ? NULL : &eventStatus[0], NULL, NULL);
			if (handled > 0)
			{
				Log(Logger::LOG_DEBUG) << __FUNC__;
				restart(timeEnd);
			}
		}
	}

	void CVODEIntegrator::assignResultsToModel()
	{
		if (mStateVector)
		{
			mModel->setStateVector(NV_DATA_S(mStateVector));
		}
	}

	void CVODEIntegrator::setCVODETolerances()
	{
		if (mStateVector == 0 || mModel == 0)
		{
			return;
		}

		// If we have a model with only events, cvode still needs a state vector of length 1 to integrate.
		if (!haveVariables() && mModel->getNumEvents() > 0)
		{
			SetVector(mStateVector, 0, 1.0);
		}

		int err = CVodeSStolerances(mCVODE_Memory, getValueAsDouble("relative_tolerance"), getValueAsDouble("absolute_tolerance"));
		if (err != CV_SUCCESS)
		{
			handleCVODEError(err);
		}

		Log(Logger::LOG_INFORMATION) << "Set tolerance to abs: " << setprecision(16)
			<< getValueAsDouble("absolute_tolerance") << ", rel: " << getValueAsDouble("relative_tolerance");
	}

	void CVODEIntegrator::restart(double time)
	{
		if (!mModel) {
			return;
		}

		// apply any events that trigger before or at time 0.
		// important NOT to set model time before we check get
		// the initial event state, initially time is < 0.
		if (time <= 0.0) {

			// copy state vector into cvode memory, need to do this before evaluating
			// roots because the applyEvents method copies the cvode state vector
			// into the model
			if (mStateVector)
			{
				mModel->getStateVector(NV_DATA_S(mStateVector));
			}

			testRootsAtInitialTime();
		}

		mModel->setTime(time);

		// copy state vector into cvode memory
		if (mStateVector && mCVODE_Memory)
		{
			mModel->getStateVector(NV_DATA_S(mStateVector));
		}

		// set tolerances and so forth.
		if (mCVODE_Memory)
		{
			reInit(time);
		}
	}

	// Cvode calls this to compute the dy/dts. This routine in turn calls the
	// model function which is located in the host application.
	int cvodeDyDtFcn(realtype time, N_Vector cv_y, N_Vector cv_ydot, void *userData)
	{
		double* y = NV_DATA_S(cv_y);
		double* ydot = NV_DATA_S(cv_ydot);
		CVODEIntegrator* cvInstance = (CVODEIntegrator*)userData;

		assert(cvInstance && "userData pointer is NULL in cvode dydt callback");

		ExecutableModel *model = cvInstance->mModel;

		model->getStateVectorRate(time, y, ydot);

		if (!cvInstance->stateVectorVariables && cvInstance->mStateVector &&
			NV_LENGTH_S(cvInstance->mStateVector) == 1)
		{
			ydot[0] = 0.0;
		}

		Log(Logger::LOG_TRACE) << __FUNC__ << ", model: " << model;

		return CV_SUCCESS;
	}

	void CVODEIntegrator::freeCVode()
	{
		// cvode does not check for null values.
		if (mCVODE_Memory)
		{
			CVodeFree(&mCVODE_Memory);
		}

		if (mStateVector)
		{
			N_VDestroy_Serial(mStateVector);
		}

		mCVODE_Memory = 0;
		mStateVector = 0;
	}

	double CVODEIntegrator::applyVariableStepPendingEvents()
	{
		if (variableStepTimeEndEvent) {
			// post event state allready calcuated.
			mModel->setStateVector(variableStepPostEventState);
			// copy state vector into cvode memory
			if (mStateVector)
			{
				mModel->getStateVector(NV_DATA_S(mStateVector));
			}
			mModel->setTime(lastEventTime);
			reInit(lastEventTime);
		}
		else {
			// apply events, copy post event status into integrator state vector.
			applyEvents(lastEventTime, eventStatus);
		}

		if (listener) {
			listener->onEvent(this, mModel, lastEventTime);
		}

		variableStepPendingEvent = false;
		variableStepTimeEndEvent = false;
		return lastEventTime;
	}

	// int (*CVRootFn)(realtype t, N_Vector y, realtype *gout, void *user_data)
	// Cvode calls this to check for event changes
	int cvodeRootFcn(realtype time, N_Vector y_vector, realtype *gout, void *user_data)
	{
		CVODEIntegrator* cvInstance = (CVODEIntegrator*)user_data;

		assert(cvInstance && "user data pointer is NULL on CVODE root callback");

		ExecutableModel *model = cvInstance->mModel;

		double* y = NV_DATA_S(y_vector);

		model->getEventRoots(time, y, gout);

		return CV_SUCCESS;
	}

	static std::string cvodeDecodeError(int cvodeError, bool exInfo)
	{
		std::string result;

		switch (cvodeError)
		{
		case CV_TOO_MUCH_WORK:
			result = "CV_TOO_MUCH_WORK";
			if (exInfo)
			{
				result += ": The solver took mxstep internal steps but "
					"could not reach tout. The default value for "
					"mxstep is MXSTEP_DEFAULT = 500.";
			}
			break;
		case CV_TOO_MUCH_ACC:
			result = "CV_TOO_MUCH_ACC";
			if (exInfo)
			{
				result += ": The solver could not satisfy the accuracy "
					"demanded by the user for some internal step.";
			}
			break;
		case CV_ERR_FAILURE:
			result = "CV_ERR_FAILURE";
			if (exInfo)
			{
				result += ": Error test failures occurred too many times "
					"(= MXNEF = 7) during one internal time step or"
					"occurred with |h| = hmin.";
			}
			break;
		case CV_CONV_FAILURE:
			result = "CV_CONV_FAILURE";
			if (exInfo)
			{
				result += ": Convergence test failures occurred too many "
					"times (= MXNCF = 10) during one internal time"
					"step or occurred with |h| = hmin.";
			}
			break;
		case CV_LINIT_FAIL:
			result = "CV_LINIT_FAIL";
			if (exInfo)
			{
				result += ": The linear solver's initialization function "
					"failed.";
			}
			break;
		case CV_LSETUP_FAIL:
			result = "CV_LSETUP_FAIL";
			if (exInfo)
			{
				result += ": The linear solver's setup routine failed in an "
					"unrecoverable manner.";
			}
			break;
		case CV_LSOLVE_FAIL:
			result = "CV_LSOLVE_FAIL";
			if (exInfo)
			{
				result += ": The linear solver's solve routine failed in an "
					"unrecoverable manner.";
			}
			break;
		case CV_RHSFUNC_FAIL:
			result = "CV_RHSFUNC_FAIL";
			break;
		case CV_FIRST_RHSFUNC_ERR:
			result = "CV_FIRST_RHSFUNC_ERR";
			break;
		case CV_REPTD_RHSFUNC_ERR:
			result = "CV_REPTD_RHSFUNC_ERR";
			break;
		case CV_UNREC_RHSFUNC_ERR:
			result = "CV_UNREC_RHSFUNC_ERR";
			break;
		case CV_RTFUNC_FAIL:
			result = "CV_RTFUNC_FAIL";
			break;
		case CV_MEM_FAIL:
			result = "CV_MEM_FAIL";
			break;
		case CV_MEM_NULL:
			result = "CV_MEM_NULL";
			if (exInfo)
			{
				result += ": The cvode_mem argument was NULL.";
			}
			break;
		case CV_ILL_INPUT:
			result = "CV_ILL_INPUT";
			if (exInfo)
			{
				result += ": One of the inputs to CVode is illegal. This "
					"includes the situation when a component of the "
					"error weight vectors becomes < 0 during "
					"internal time-stepping.  It also includes the "
					"situation where a root of one of the root "
					"functions was found both at t0 and very near t0. "
					"The ILL_INPUT flag will also be returned if the "
					"linear solver routine CV--- (called by the user "
					"after calling CVodeCreate) failed to set one of "
					"the linear solver-related fields in cvode_mem or "
					"if the linear solver's init routine failed. In "
					"any case, the user should see the printed "
					"error message for more details.";
			}
			break;
		case CV_NO_MALLOC:
			result = "CV_NO_MALLOC";
			if (exInfo)
			{
				result += ": indicating that cvode_mem has not been "
					"allocated (i.e., CVodeInit has not been "
					"called).";
			}
			break;
		case CV_BAD_K:
			result = "CV_BAD_K";
			if (exInfo)
			{
				result += ": k is not in the range 0, 1, ..., qu.";
			}
			break;
		case CV_BAD_T:
			result = "CV_BAD_T";
			if (exInfo)
			{
				result += ": t is not in the interval [tn-hu,tn].";
			}
			break;
		case CV_BAD_DKY:
			result = "CV_BAD_DKY";
			if (exInfo)
			{
				result += ": The dky argument was NULL.";
			}
			break;
		case CV_TOO_CLOSE:
			result = "CV_TOO_CLOSE:";
			break;
		default:
			result = "UNKNOWN_CODE";
			break;
		}
		return result;
	}


	/**
	* Purpose
	* This function processes error and warning messages from CVODE and its
	* sub-modules.
	*
	* Arguments:
	* error_code: is the error code.
	* module: is the name of the CVODE module reporting the error.
	* function: is the name of the function in which the error occurred.
	* msg: is the error message.
	* eh_data: is a pointer to user data, the same as the eh_data parameter
	* passed to CVodeSetErrHandlerFn.
	* Return value: A CVErrHandlerFn function has no return value.
	*
	* Notes:
	* error_code is negative for errors and positive (CV_WARNING) for warnings.
	* If a function that returns a pointer to memory encounters an error,
	* it sets error_code to 0.
	*/
	void cvodeErrHandler(int error_code, const char *module, const char *function,
		char *msg, void *eh_data)
	{
		if (error_code < 0) {
			Log(Logger::LOG_ERROR) << "CVODE Error: " << cvodeDecodeError(error_code, false)
				<< ", Module: " << module << ", Function: " << function
				<< ", Message: " << msg;

		}
		else if (error_code == CV_WARNING) {
			Log(Logger::LOG_WARNING) << "CVODE Warning: "
				<< ", Module: " << module << ", Function: " << function
				<< ", Message: " << msg;
		}
	}

}


