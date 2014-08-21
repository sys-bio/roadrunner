/*
 * PyLoggerStream.cpp
 *
 *  Created on: Aug 13, 2014
 *      Author: andy
 */

#include <PyLoggerStream.h>


namespace rr
{

class PyGILState
{
public:
    PyGILState_STATE state;

    PyGILState() : state(PyGILState_Ensure()) {};

    ~PyGILState() {  PyGILState_Release(state); };
};



PyLoggerStream::~PyLoggerStream()
{
    freePyObjects();
}



int PyLoggerStream::sync()
{
    PyGILState gstate;

    PyObject* args = PyTuple_New(0);

    PyObject *retval = PyObject_Call(flush, args, NULL);

    Py_DECREF(args);

    if (retval) {
        if (retval == Py_None) {
            return 1;
        }
        else {
            throw std::runtime_error("error calling python flush method, result not Py_None");
            return -1;
        }
    }

    throw std::runtime_error("error calling python flush method, result NULL");

    return -1;
}

// write one character
int PyLoggerStream::overflow(int c) {

    if (c != EOF) {

        char buffer[2] = {0};
        buffer[0] = (char)c;
        buffer[1] = 0;
        if (xsputn(buffer, 1) != 1) {
            return EOF;
        }
    }

    return c;
}

std::streamsize PyLoggerStream::xsputn (const char* s,
        std::streamsize num) {

    PyGILState gstate;

    if (!stream) {
        throw (std::runtime_error("Python stream object is NULL, "
                "likely PyLoggerStream has not been initialized"));
    }

    // new python object
    PyObject* pystr = PyString_FromStringAndSize(s, num);

    // new py object
    PyObject* args = PyTuple_New(1);

    // steals ref to pystr
    PyTuple_SetItem(args, 0, pystr);

    PyObject *retval = PyObject_Call(write, args, NULL);

    Py_DECREF(args);

    int result = EOF;

    if (retval) {
        if (retval == Py_None) {
            result = num;
        }
        else {
            throw std::runtime_error("error calling python write method, result not Py_None");
            result = EOF;
        }
    } else {

        throw std::runtime_error("error calling python write method, result NULL");
    }

    return result;
}

// single global pyLogger object does not allocate any memory,
// holds onto pointers created in python.
static PyLoggerStream pyLoggerStream;

// references the above stram.
static std::ostream pyostream(&pyLoggerStream);


void PyLoggerStream::enablePythonLogging()
{
    PyGILState gstate;

    // a borrowed reference to python owned object,
    // the logger stream increments the refe count.
    PyObject* pyerr = PySys_GetObject("stderr");

    if (pyerr == NULL)
    {
        throw std::logic_error("ERROR: Could not get Python stderr");
        return;
    }

    pyLoggerStream.setPyStream(pyerr);
    Logger::setConsoleStream(&pyostream);
}


void PyLoggerStream::disablePythonLogging()
{
    PyGILState gstate;

    Logger::setConsoleStream(&std::clog);

    pyLoggerStream.freePyObjects();
}

PyLoggerStream::PyLoggerStream(const PyLoggerStream& other):
    stream(NULL), write(NULL), flush(NULL)
{
    throw std::logic_error("object is not copyable");
}

PyLoggerStream& PyLoggerStream::operator =(const PyLoggerStream&)
{
    throw std::logic_error("object is not copyable");
}

PyLoggerStream::PyLoggerStream():
        stream(NULL), write(NULL), flush(NULL)
{
}


void PyLoggerStream::setPyStream(PyObject* ps)
{
    freePyObjects();

    if (ps == NULL) {
        return;
    }

    try {
        stream = ps;
        Py_INCREF(stream);

        write = PyObject_GetAttrString(stream, "write");
        if (write == NULL) {
            throw std::invalid_argument("given python object has no write method");
        }

        if (!PyCallable_Check(write)) {
            throw std::invalid_argument("write attribute of python stream object is not callable");
        }

        flush = PyObject_GetAttrString(stream, "flush");
        if (flush == NULL) {
            throw std::invalid_argument("given python object has no flush method");
        }

        if (!PyCallable_Check(flush)) {
            throw std::invalid_argument("flush attribute of python stream object is not callable");
        }
    } catch(std::exception& e) {
        freePyObjects();
        throw;
    }
}

void PyLoggerStream::freePyObjects()
{
    if(Py_IsInitialized()) {
        PyGILState gstate;

        if(stream) {
            Py_DECREF(stream);
            stream = NULL;
        }

        if(write) {
            Py_DECREF(write);
            write = NULL;
        }

        if(flush) {
            Py_DECREF(flush);
            flush = NULL;
        }
    } else {
        std::cout << "python not initialized " << std::endl;
        stream = NULL;
        write = NULL;
        flush = NULL;
    }
}

} /* namespace rr */
