/*
 * PyLoggerStream.h
 *
 *  Created on: Aug 13, 2014
 *      Author: andy
 */

#ifndef PYLOGGERSTREAM_H_
#define PYLOGGERSTREAM_H_

#include <cstdio>
#include <streambuf>

#include <iostream>
#include <stdexcept>
#include "rrLogger.h"


#include <Python.h>


namespace rr {


/**
 * Object that implements the streambuff interface around the
 * an existing FILE object that was created by python.
 */
class PyLoggerStream : public std::streambuf
{
public:

    /**
     * initialize a static PyLoggerStream object with
     * the Python sys.stderr object and attach it
     * to the roadrunner Logger.
     */
    static void enablePythonLogging();


    static void disablePythonLogging();

    PyLoggerStream();


    virtual ~PyLoggerStream();

    PyLoggerStream( const PyLoggerStream& other ); // non construction-copyable
    PyLoggerStream& operator=( const PyLoggerStream& ); // non copyable


    /**
     * called by ostream
     */
    void close();

    /**
     *  @brief  Consumes data from the buffer; writes to the
     *          controlled sequence.
     *  @param  c  An additional character to consume.
     *  @return  eof() to indicate failure, something else (usually
     *           @a c, or not_eof())
     *
     *  Called by ostream
     *
     *  writes a single character to the python stream.
     *
     *  Informally, this function is called when the output buffer is full
     *  (or does not exist, as buffering need not actually be done).  If a
     *  buffer exists, it is "consumed", with "some effect" on the
     *  controlled sequence.  (Typically, the buffer is written out to the
     *  sequence verbatim.)  In either case, the character @a c is also
     *  written out, if @a c is not @c eof().
     *
     *  For a formal definiton of this function, see a good text
     *  such as Langer & Kreft, or [27.5.2.4.5]/3-7.
     *
     *  A functioning output streambuf can be created by overriding only
     *  this function (no buffer area will be used).
     *
     *  @note  Base class version does nothing, returns eof().
     */
    virtual int overflow(int c);

    /**
     * called by ostream
     *
     * Synchronizes the buffer arrays with the controlled sequences.
     *
     * tell py stream to flush
     *
     * @return  -1 on failure.
     */
    virtual int sync();

    /**
     * write multiple chars,
     * called by C++ ostream
     *
     *  @param  s  A buffer area.
     *  @param  n  Maximum number of characters to write.
     *  @return  The number of characters written.
     *
     *  Writes @a s[0] through @a s[n-1] to the output sequence, as if
     *  by @c sputc().  Stops when either @a n characters have been
     *  copied, or when @c sputc() would return @c traits::eof().
     *
     *  It is expected that derived classes provide a more efficient
     *  implementation by overriding this definition.
     */
    virtual std::streamsize xsputn (const char* s,
            std::streamsize num) ;

private:

    /**
     * set the python objects from a python stream object.
     *
     * This object holds onto the py stream, so the ref count is incremented.
     */
    void setPyStream(PyObject* ps);

    void freePyObjects();

    // reference to python stream object, incr when getting, decr when free
    PyObject *stream;

    // reference to write method of stream object, incr when getting, decr when free
    PyObject *write;

    // reference to flush method of stream object, incr when getting, decr when free
    PyObject *flush;
};

} /* namespace rr */

#endif /* PYLOGGERSTREAM_H_ */
