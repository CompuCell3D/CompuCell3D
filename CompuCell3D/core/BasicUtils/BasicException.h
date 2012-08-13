/*******************************************************************\

              Copyright (C) 2003 Joseph Coffland

    This program is free software; you can redistribute it and/or
     modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; either version 2
        of the License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
             GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
     along with this program; if not, write to the Free Software
      Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
                           02111-1307, USA.

            For information regarding this software email
                   jcofflan@users.sourceforge.net

\*******************************************************************/


#ifndef BASICEXCEPTION_H
#define BASICEXCEPTION_H

#include "BasicFileLocation.h"
#include "BasicSmartPointer.h"
#include "Zap.h"

#include <string>
#include <iostream>
#include <list>

// Forward Declarations
template <class T, sp_alloc_t alloc_t>
class BasicSmartPointer;

/**
 * BasicException is a general purpose exception class.  It is similar to
 * the java Exception class.  A BasicException can carry a message,
 * a BasicFileLocation and/or a pointer to an exception which was the
 * original cause.
 *
 * There are preprocessor macros that can be used to as a convient way
 * to add the current file, line and column where the exception occured.
 * These are:
 *
 * THROW(const string message)
 * 
 * and
 * 
 * ASSERT_OR_THROW(const string message, const bool condition)
 *
 * The latter can be used to in place of assert(const bool condition).  
 * Throwing an exception instead of aborting overcomes some of the limitations 
 * of the standard assert.
 */
class BasicException {
  std::string message;
  BasicFileLocation location;
  BasicSmartPointer<BasicException> cause;
  BasicSmartPointer<std::list<std::string> > trace;

public:
  static unsigned int causePrintLevel;
  static bool enableStackTraces;

  BasicException() {init();}

  BasicException(const std::string message) : message(message) {
    init();
  }

  BasicException(const std::string message, const BasicFileLocation &location) 
    : message(message), location(location) {
    init();
  }
  
  BasicException(const std::string message, BasicException &cause) :
    message(message) {
    this->cause = new BasicException(cause);
    init();
  }

  BasicException(const std::string message, const BasicFileLocation &location, 
		 BasicException &cause) :
    message(message), location(location) {
    this->cause = new BasicException(cause);
    init();
  }

  /// Copy constructor
  BasicException(const BasicException &e) :
    message(e.message), location(e.location), cause(e.cause), trace(e.trace) {}

  virtual ~BasicException() {}

  const std::string getMessage() const {return message;}
  BasicFileLocation getLocation() const {return location;}

  /**
   * @return A BasicSmartPointer to the BasicException that caused this 
   *         exception or NULL.
   */  
  BasicSmartPointer<BasicException> getCause() const {return cause;}

  BasicSmartPointer<std::list<std::string> > getTrace() const {return trace;}

  /** 
   * Prints the complete exception recuring down to the cause exception if
   * not null.  WARNING: If there are many layers of causes this function
   * could print a very large amount of data.  This can be limited by
   * setting the causePrintLevel variable.
   * 
   * @param stream The output stream.
   * @param printLocations Print file locations.
   * @param printLevel The current cause print level.
   * 
   * @return A reference to the passed stream.
   */  
  std::ostream &print(std::ostream &stream,
		      bool printLocations = true,
		      unsigned int printLevel = 0) const {

    if (printLocations && !location.isEmpty())
      stream << "@ " << location << " ";

    stream << message;

    if (enableStackTraces && !trace.isNull()) {
      std::list<std::string>::iterator it;
      for (it = trace->begin(); it != trace->end(); it++)
	stream << std::endl << "  " << *it;
    }

    if (!cause.isNull()) {
      stream << std::endl << " ";

      if (printLevel > causePrintLevel) {
	stream << "Aborting exception dump due to causePrintLevel limit! "
	       << "Increase BasicException::causePrintLevel to see more.";

      } else {
	stream << "caused by: ";
	cause->print(stream, printLocations, printLevel);
      }
    }

    return stream;
  }

protected:
  void init() {
    if (enableStackTraces) {
      trace = new std::list<std::string>;

      // When Optimization is turned on functions such as this
      // one are often inlined and not visable to the debugger.
      // This means stack traces for optimized code will often
      // be incomplete.  Here we remove the offset in order
      // to get as much of the stack trace as possible.
    }
  }

  friend std::ostream &operator<<(std::ostream &, const BasicException &);
};

/** 
 * An stream output operator for BasicException.  This allows you to print the
 * text of an exception to a stream like so:
 *
 * . . .
 * } catch (BasicException &e) {
 *   cerr << e << endl;
 *   return 0;
 * }
 */
inline std::ostream &operator<<(std::ostream &stream,
				const BasicException &e) {
  e.print(stream);
  return stream;
}

#define THROW(msg) throw BasicException((msg), FILE_LOCATION)
#define THROWC(msg, cause) throw BasicException((msg), FILE_LOCATION, (cause))
#define ASSERT_OR_THROW(msg, condition) {if (!(condition)) THROW(msg);}

#endif
