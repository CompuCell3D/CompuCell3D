/*
  Last changed Time-stamp: <2008-10-16 19:11:11 raim>
  $Id: solverError.c,v 1.24 2008/11/07 09:15:46 raimc Exp $ 
*/
/* 
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation; either version 2.1 of the License, or
 * any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY, WITHOUT EVEN THE IMPLIED WARRANTY OF
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE. The software and
 * documentation provided hereunder is on an "as is" basis, and the
 * authors have no obligations to provide maintenance, support,
 * updates, enhancements or modifications.  In no event shall the
 * authors be liable to any party for direct, indirect, special,
 * incidental or consequential damages, including lost profits, arising
 * out of the use of this software and its documentation, even if the
 * authors have been advised of the possibility of such damage.  See
 * the GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
 *
 * The original code contained here was initially developed by:
 *
 *     Andrew M. Finney
 *
 * Contributor(s):
 */

/*! \defgroup errors SOSlib Error Management
  \brief This optional module contains all functions to set and
  retrieve warnings, errors and fatal errors occuring at any
  level of SOSlib.

*/
/*@{*/

#include "sbmlsolver/solverError.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#ifdef WIN32
#define vsnprintf _vsnprintf
#include "windows.h"
#endif

#include <sbml/util/List.h>

/** error message, including errorCode */
typedef struct solverErrorMessage
{
  char *message ;
  int errorCode ;
} solverErrorMessage_t ;

static int SolverError_dumpHelper(char *);

/*!!! TODO : GLOBAL ERROR LIST - remove and add separate lists
  for each internal structure !!!*/
static List_t *solverErrors[NUMBER_OF_ERROR_TYPES] = { NULL, NULL, NULL, NULL };

static int memoryExhaustion = 0;
static solverErrorMessage_t memoryExhaustionFixedMessage =
  {
    "No more memory avaliable",
    SOLVER_ERROR_NO_MORE_MEMORY_AVAILABLE
  };

/* get number of stored errors  of given type */
SBML_ODESOLVER_API int SolverError_getNum(errorType_t type)
{
  List_t *errors = solverErrors[type];

  return (errors ? List_size(errors) : 0) +
    (type == FATAL_ERROR_TYPE ? memoryExhaustion : 0) ;
}

SBML_ODESOLVER_API solverErrorMessage_t *SolverError_getError(errorType_t type, int errorNum)
{
  List_t *errors = solverErrors[type];

  if ( type == FATAL_ERROR_TYPE && memoryExhaustion &&
       errorNum == (errors ? List_size(errors) : 0) )
    return &memoryExhaustionFixedMessage ;

  if ( !errors )
    return NULL;

  return List_get(errors, errorNum);
}

/** get a stored error message */
SBML_ODESOLVER_API char *SolverError_getMessage(errorType_t type, int errorNum)
{
  return SolverError_getError(type, errorNum)->message ;
}

/** get error code */
SBML_ODESOLVER_API errorCode_t SolverError_getCode(errorType_t type, int errorNum)
{
  return SolverError_getError(type, errorNum)->errorCode ; 
}

/** get error code of last error stored of given type */
SBML_ODESOLVER_API errorCode_t SolverError_getLastCode(errorType_t type)
{
  if ( !SolverError_getNum(type) )
    return 0;
  else
    return SolverError_getCode(type, SolverError_getNum(type) - 1);
}

/** empty error store */
SBML_ODESOLVER_API void SolverError_clear()
{
  int i ;

  for ( i = 0; i != NUMBER_OF_ERROR_TYPES; i++ )
  {
    List_t *l = solverErrors[i];

    if ( l )
    {
      while ( List_size(l) )
      {
	solverErrorMessage_t *m = List_get(l, 0);

	free(m->message);
	free(m);
	List_remove(l, 0);
      }      
    }
    /* List_free(l); */ /* RM: removed again, causes seg.fault should be done elsewwhere ?*/
  }

  memoryExhaustion = 0;
}

SBML_ODESOLVER_API void SolverError_dumpAndClearErrors()
{
  SolverError_dump();
  SolverError_clear();
}


/** create an error */
SBML_ODESOLVER_API void SolverError_error(errorType_t type, errorCode_t errorCode, char *fmt, ...)
{
  List_t *errors = solverErrors[type];
  char buffer[2000], *variableLengthBuffer;
  va_list args;
  solverErrorMessage_t *message =
    (solverErrorMessage_t *)malloc(sizeof(solverErrorMessage_t));

  if ( message == NULL )
    memoryExhaustion = 1;
  else
  {
    va_start(args, fmt);
    vsnprintf(buffer, 2000, fmt, args);
    va_end(args);

    variableLengthBuffer = (char *)malloc(strlen(buffer) + 1);
    message->errorCode = errorCode;

    if ( variableLengthBuffer == NULL )
      memoryExhaustion = 1;
    else
    {
      message->message = strcpy(variableLengthBuffer, buffer);

      if ( !errors )
	errors = solverErrors[type] = List_create();

      List_add(errors, message);
    }
  }
}

/** exit the program if errors or fatals have been created. */
SBML_ODESOLVER_API void SolverError_haltOnErrors()
{
  if ( SolverError_getNum(ERROR_ERROR_TYPE) ||
       SolverError_getNum(FATAL_ERROR_TYPE) )
    exit(EXIT_FAILURE);
}


/** write all errors and warnings to a string (owned by caller
    unless memoryExhaustion) */
SBML_ODESOLVER_API char *SolverError_dumpToString()
{
  char *result;

  /*AIX: deactivate memoryExhaustion, this is a hack required
    under AIX because 'static int memoryExhaustion=0' does not
    actually initialize it to 0 */
#if defined(_AIX) || defined(__AIX) || defined(__AIX__) || defined(__aix) || defined(__aix__)
  memoryExhaustion = 0;
#endif
    
  if ( !memoryExhaustion )
  {
    int bufferSize = SolverError_dumpHelper(NULL);
    result = SolverError_calloc(bufferSize, sizeof(char *));
  }

  if ( memoryExhaustion )
    result = "Fatal Error\t30000\tNo more memory avaliable\n";
  else
    SolverError_dumpHelper(result);

  return result;
}

/* free string returned by SolverError_dumpToString */
SBML_ODESOLVER_API void SolverError_freeDumpString(char *message)
{
  if ( !memoryExhaustion )
    free(message);
}

/* write all errors and warnings to standard error */
SBML_ODESOLVER_API void SolverError_dump()
{
  char *message = SolverError_dumpToString();

  fprintf(stderr, "%s", message);
  SolverError_freeDumpString(message);
}

/* returns 1 if memory has been exhausted 0 otherwise */
SBML_ODESOLVER_API int SolverError_isMemoryExhausted()
{
  return memoryExhaustion;
}

SBML_ODESOLVER_API void *SolverError_calloc(size_t num, size_t size)
{
  /* static int noOfCalls = 0; for testing */
  void *result;

  /*noOfCalls++;

  if (noOfCalls > 1)
  result = NULL ;
  else */
    
  result = calloc(num, size);

  memoryExhaustion = memoryExhaustion || !result ;

  /*AIX: deactivate memoryExhaustion, this is a hack required
    under AIX because 'static int memoryExhaustion=0' does not
    actually initialize it to 0 */
#if defined(_AIX) || defined(__AIX) || defined(__AIX__) || defined(__aix) || defined(__aix__)
  memoryExhaustion = 0;
#endif
    
  return result ;
}

#ifdef WIN32
void SolverError_storeLastWin32Error(const char *context)
{
  LPVOID lpMsgBuf;

  if ( !FormatMessage(
		      FORMAT_MESSAGE_ALLOCATE_BUFFER |
		      FORMAT_MESSAGE_FROM_SYSTEM |
		      FORMAT_MESSAGE_IGNORE_INSERTS,
		      NULL,
		      GetLastError(),
		      /* Default language */
		      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
		      (LPTSTR) &lpMsgBuf,
		      0,
		      NULL ) )
  {
    SolverError_error(FATAL_ERROR_TYPE, SOLVER_ERROR_WIN32_FORMAT_ERROR,
		      "Fatal Error cause unknown - (FormatMessage failed)\n");
    return ;
  }

  SolverError_error(ERROR_ERROR_TYPE, SOLVER_ERROR_WIN32_ERROR,
		    "%s - %s", context, (const char *)lpMsgBuf);
    
  // Free the buffer.
  LocalFree( lpMsgBuf );
}
#endif

/** @} */
/* our portable clone of itoa */
char* SolverError_itoa( int value, char* result, int base )
{
  char *out = result, *reverseSource, *reverseTarget;
  int quotient = value;

  /* check that the base if valid */
  if ( base < 2 || base > 16 ) { *result = 0; return result; }

  do
  {
    *out = "0123456789abcdef"[ abs( quotient % base ) ];
    ++out;
    quotient /= base;
  }
  while ( quotient );

  if ( value < 0 ) *out++ = '-';

  reverseTarget = result ;
  reverseSource = out;

  while ( reverseSource > reverseTarget )
  {
    char temp;

    reverseSource--;
    temp = *reverseSource ;
    *reverseSource = *reverseTarget;
    *reverseTarget = temp ;
    reverseTarget++;
  }

  *out = 0;
  return result;
}


static int SolverError_dumpHelper(char *s)
{
  int result = 1;

  static char *solverErrorTypeString[] =
    { "Fatal Error",
      "      Error",
      "    Warning",
      "    Message" } ;

  int i, j;

  for ( i=0; i != NUMBER_OF_ERROR_TYPES; i++ )
  {
    List_t *errors = solverErrors[i];

    if ( errors )
    {
      for ( j=0; j != List_size(errors); j++ )
      {
	char errorCodeString[35] ;
	solverErrorMessage_t *error = List_get(errors, j);

	SolverError_itoa(error->errorCode, errorCodeString, 10);
                    
	if ( s )
	{
	  result = sprintf(s, "%s\t%s\t%s\n",
			   solverErrorTypeString[i],
			   errorCodeString, error->message);
	  s += result ;
	}
	else
	  result +=
	    3 +
	    strlen(solverErrorTypeString[i]) +
	    strlen(error->message) +
	    strlen(errorCodeString);
      }
    }
  }

  if ( s )
    *s = '\0';

  return result ;
}
