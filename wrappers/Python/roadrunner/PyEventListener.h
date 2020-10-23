#ifndef PyEventHandler_H_
#define PyEventHandler_H_

#include "rrExecutableModel.h"
#include "rrLogger.h"

namespace rr
{
class PyEventListener : public EventListener
{
public:

    PyEventListener() : pyOnTrigger(0), pyOnAssignment(0)
    {
        Log(Logger::LOG_INFORMATION) << __FUNC__;
    }

    virtual ~PyEventListener()
    {
        Log(Logger::LOG_NOTICE) << __FUNC__;

        PyGILState_STATE gstate = PyGILState_Ensure();

        Py_XDECREF(pyOnTrigger);
        Py_XDECREF(pyOnAssignment);

        PyGILState_Release(gstate);
    }

    void setOnTrigger(PyObject *py)
    {
		if (py == Py_None) {
			Py_XDECREF(pyOnTrigger);  /* Dispose of previous callback */
			pyOnTrigger = NULL;
			return;
		}

        if (!PyCallable_Check(py)) {
            throw std::invalid_argument("argument must be callable");
        }

        Py_XINCREF(py);           /* Add a reference to new callback */
        Py_XDECREF(pyOnTrigger);  /* Dispose of previous callback */
        pyOnTrigger = py;         /* Remember new callback */
    }

    PyObject* getOnTrigger()
    {
		if (pyOnTrigger) {
			return pyOnTrigger;
		} else {
			Py_RETURN_NONE;
		}
    }

    void setOnAssignment(PyObject* py)
    {
		if (py == Py_None) {
			Py_XDECREF(pyOnAssignment);  /* Dispose of previous callback */
			pyOnAssignment = NULL;
			return;
		}

        if (!PyCallable_Check(py)) {
            throw std::invalid_argument("argument must be callable");
        }

        Py_XINCREF(py);              /* Add a reference to new callback */
        Py_XDECREF(pyOnAssignment);  /* Dispose of previous callback */
        pyOnAssignment = py;           /* Remember new callback */

    }

    PyObject *getOnAssignment()
    {
		if (pyOnAssignment) {
			return pyOnAssignment;
		} else {
			Py_RETURN_NONE;
		}
    }

    void fireOnTrigger(ExecutableModel* model, size_t index, const std::string& eventId)
    {
        Log(Logger::LOG_INFORMATION) << __FUNC__;
        onTrigger(model, index, eventId);
    }

    void fireOnAssignment(ExecutableModel* model, size_t index, const std::string& eventId)
    {
        Log(Logger::LOG_INFORMATION) << __FUNC__;
        onAssignment(model, index, eventId);
    }

private:

    PyObject *pyOnTrigger;
    PyObject *pyOnAssignment;

    virtual uint onTrigger(ExecutableModel* model, size_t index, const std::string& eventId)
    {
        uint result = 0;
        std::string err;

        Log(Logger::LOG_INFORMATION) << __FUNC__ << "model: " << (void*)model << ", event id: " << eventId << ", index: " << index;

        if (pyOnTrigger) {

            PyGILState_STATE gstate = PyGILState_Ensure();

            PyObject *pyModel = ExecutableModel_NewPythonObj(model);

            /* format strings:
               N (object) [PyObject *] - Same as O, except it doesn’t increment the reference count
               on the object. Useful when the object is created by a call to an object constructor
               in the argument list.
            */

            PyObject *args = Py_BuildValue("(N, i, s)", pyModel, index, eventId.c_str());
            PyObject *pyres = PyEval_CallObject(pyOnTrigger, args);


            if (PyErr_Occurred()) {

                PyObject* pystr = PyObject_Str(PyErr_Occurred());
                err = std::string("Error calling Python onTrigger method: ") + PyString_AsString(pystr);

                Log(Logger::LOG_ERROR) << err;

                Py_XDECREF(pystr);
                PyErr_Clear();
            }
            else if (PyInt_Check(pyres)) {
                result = PyInt_AsLong(pyres);
            }
            else if (pyres != Py_None) {
                PyObject* pystr = PyObject_Str(pyres);
                const char* cstr = PyString_AsString(pystr);

                Log(Logger::LOG_WARNING) << "The Python onTrigger handler returned " << cstr;
                Log(Logger::LOG_WARNING) << "None or an integer are the only valid return values";

                Py_XDECREF(pystr);
            }

            Py_XDECREF(pyres);
            Py_XDECREF(args);

            // Release the thread. No Python wrappers allowed beyond this point.
            PyGILState_Release(gstate);

            if (!err.empty())
            {
                throw std::runtime_error(err);
            }
        }

        return result;
    }

    virtual uint onAssignment(ExecutableModel* model, size_t index, const std::string& eventId)
    {
        uint result = 0;
        std::string err;

        Log(Logger::LOG_INFORMATION) << __FUNC__ << "model: " << (void*)model << ", event id: " << eventId << ", index: " << index;

        if (pyOnAssignment) {
            PyGILState_STATE gstate = PyGILState_Ensure();

            PyObject *pyModel = ExecutableModel_NewPythonObj(model);
            PyObject *args = Py_BuildValue("(N, i, s)", pyModel, index, eventId.c_str());
            PyObject *pyres = PyEval_CallObject(pyOnAssignment, args);

            if (PyErr_Occurred()) {

                PyObject* pystr = PyObject_Str(PyErr_Occurred());
                err = std::string("Error calling Python onAssignment method: ") + PyString_AsString(pystr);

                Log(Logger::LOG_ERROR) << err;

                Py_XDECREF(pystr);
                PyErr_Clear();
            }
            else if (PyInt_Check(pyres)) {
                result = PyInt_AsLong(pyres);
            }
            else if (pyres != Py_None) {
                PyObject* pystr = PyObject_Str(pyres);
                const char* cstr = PyString_AsString(pystr);

                Log(Logger::LOG_WARNING) << "The Python onAssignment handler returned " << cstr;
                Log(Logger::LOG_WARNING) << "None or an integer are the only valid return values";

                Py_XDECREF(pystr);
            }


            Py_XDECREF(pyres);
            Py_XDECREF(args);

            // Release the thread. No Python wrappers allowed beyond this point.
            PyGILState_Release(gstate);

            if (!err.empty())
            {
                throw std::runtime_error(err);
            }
        }

        return result;
    }
};


}

#endif // PyEventHandler_H_
