/*
  Last changed Time-stamp: <2008-10-06 15:01:44 raim>
  $Id: solverError.h,v 1.33 2008/10/08 17:07:16 raimc Exp $ 
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
 *     Andrew Finney
 *
 * Contributor(s):
 *     Rainer Machne
 */

#ifndef _SOLVERERROR_H_
#define _SOLVERERROR_H_

#include <stdarg.h>
#include <stddef.h>

#include "sbmlsolver/exportdefs.h"

/** error codes.
    codes < 0 reserved for CVODE\n
    codes 0 throu 99999 reserved for LibSBML\n
    codes 0 throu 9999 libSBML, XML layer\n
    codes 10000 throu 99999 libSBML, SBML Errors\n */
enum errorCode
  {
    
    /** 10XXXX - conversion to ode model failures in odeConstruct.c */
    SOLVER_ERROR_ODE_COULD_NOT_BE_CONSTRUCTED_FOR_SPECIES  = 100000,
    SOLVER_ERROR_THE_MODEL_CONTAINS_EVENTS = 100001,
    SOLVER_ERROR_THE_MODEL_CONTAINS_ALGEBRAIC_RULES = 100002,
    SOLVER_ERROR_ODE_MODEL_COULD_NOT_BE_CONSTRUCTED = 100003,
    SOLVER_ERROR_NO_KINETIC_LAW_FOUND_FOR_REACTION = 100004,
    SOLVER_ERROR_ENTRIES_OF_THE_JACOBIAN_MATRIX_COULD_NOT_BE_CONSTRUCTED = 100005,
    SOLVER_ERROR_MODEL_NOT_SIMPLIFIED = 100006,
    SOLVER_ERROR_ENTRIES_OF_THE_PARAMETRIC_MATRIX_COULD_NOT_BE_CONSTRUCTED = 100007,
    SOLVER_ERROR_REQUESTED_PARAMETER_NOT_FOUND = 100008,
    SOLVER_ERROR_THE_MODEL_CONTAINS_PIECEWISE = 100009,
    SOLVER_ERROR_ODE_MODEL_CYCLIC_DEPENDENCY_IN_RULES = 100010,
    SOLVER_ERROR_ODE_MODEL_RULE_SORTING_FAILED = 100011,
    SOLVER_ERROR_ODE_MODEL_SET_DISCONTINUITIES_FAILED = 100012,

    /** 11xx30 - SBML input model failures in sbml.c */
    SOLVER_ERROR_MAKE_SURE_SCHEMA_IS_ON_PATH = 110030,
    SOLVER_ERROR_CANNOT_PARSE_MODEL = 110031,
    SOLVER_ERROR_DOCUMENTLEVEL_ONE = 1100032,
     
    /** 11XX5X - Graph Drawing Errors  in drawGraph.c */
    SOLVER_ERROR_NO_GRAPHVIZ = 110050,
    /** 11X1XX - Wrong Input Settings */
    SOLVER_ERROR_INTEGRATOR_SETTINGS = 110100,
    SOLVER_ERROR_VARY_SETTINGS = 110101,
      
    /** 12XXXX - Integration Failures in integratorInstance.c */
    SOLVER_ERROR_INTEGRATION_NOT_SUCCESSFUL = 120000,
    SOLVER_ERROR_EVENT_TRIGGER_FIRED = 120001,
    SOLVER_ERROR_CVODE_MALLOC_FAILED = 120002,
    SOLVER_ERROR_CVODE_REINIT_FAILED = 120003,

    /** 12X1XX - AST Processing Failures in processAST.c */
    /** AST evaluation in evaluateAST */
    SOLVER_ERROR_AST_UNKNOWN_NODE_TYPE = 120100,
    SOLVER_ERROR_AST_UNKNOWN_FAILURE = 120101,
    SOLVER_ERROR_AST_EVALUATION_FAILED_MISSING_VALUE = 120102,
    SOLVER_ERROR_AST_EVALUATION_FAILED_DELAY = 120103,
    SOLVER_ERROR_AST_EVALUATION_FAILED_LAMBDA = 120104,
    SOLVER_ERROR_AST_EVALUATION_FAILED_FUNCTION = 120105,
    SOLVER_ERROR_AST_EVALUATION_FAILED_FLOAT_FACTORIAL = 120106,
    SOLVER_ERROR_AST_EVALUATION_FAILED_PIECEWISE = 120107,
    SOLVER_ERROR_AST_EVALUATION_FAILED_DISCRETE_DATA = 120108,
    /** AST differentiation in differentiateAST */
    SOLVER_ERROR_AST_DIFFERENTIATION_FAILED_CONSTANT = 120110,
    SOLVER_ERROR_AST_DIFFERENTIATION_FAILED_OPERATOR = 120111,
    SOLVER_ERROR_AST_DIFFERENTIATION_FAILED_LAMBDA = 120112,
    SOLVER_ERROR_AST_DIFFERENTIATION_FAILED_DELAY = 120114,
    SOLVER_ERROR_AST_DIFFERENTIATION_FAILED_FACTORIAL = 120115,
    SOLVER_ERROR_AST_DIFFERENTIATION_FAILED_PIECEWISE = 120117,
    SOLVER_ERROR_AST_DIFFERENTIATION_FAILED_LOGICAL_OR_RELATIONAL = 120118,
    
    /** 12X2XX - Result Writing Failures */
    SOLVER_ERROR_CVODE_RESULTS_FAILED = 120201,
    SOLVER_ERROR_SBML_RESULTS_FAILED = 120202,

    /** 12X3XX - Adjoint Solver Failures */
       
    /** 12X4XX - Objective Function Failures */      
    SOLVER_ERROR_VECTOR_V_FAILED = 120401,
    SOLVER_ERROR_OBJECTIVE_FUNCTION_FAILED = 120402,
    SOLVER_ERROR_OBJECTIVE_FUNCTION_V_VECTOR_COULD_NOT_BE_CONSTRUCTED = 120403,
      
    /** 12X5XX - Integration Messages in integratorInstance.c */
    SOLVER_MESSAGE_RERUN_WITH_OR_WO_JACOBIAN = 120500,
    SOLVER_MESSAGE_STEADYSTATE_FOUND = 120501,
    SOLVER_ERROR_UPDATE_ADJDATA = 120502, 
    SOLVER_ERROR_INITIALIZE_ADJDATA = 120503,

    /** 12X6XX - Errors and Messages in input data */
    SOLVER_MESSAGE_INTERPOLATION_OUT_OF_RANGE = 120600,
      
    /** 13XXXX - Memory Exhaustion; general */
    SOLVER_ERROR_NO_MORE_MEMORY_AVAILABLE = 130000,

    /** 13025X - Win32 Errors */
    SOLVER_ERROR_WIN32_ERROR = 130250,
    SOLVER_ERROR_WIN32_FORMAT_ERROR = 130251,

    /** 1305XX - Compilation Errors */
    SOLVER_ERROR_COMPILATION_FAILED = 130500,
    SOLVER_ERROR_CANNOT_COMPILE_JACOBIAN_NOT_COMPUTED = 130501,
    SOLVER_ERROR_AST_COMPILATION_FAILED_DATA_AST_NODE_NOT_SUPPORTED_YET = 130502,
    SOLVER_ERROR_AST_COMPILATION_FAILED_MISSING_VALUE = 130503,
    SOLVER_ERROR_AST_COMPILATION_FAILED_STRANGE_NODE_TYPE = 130504,
    SOLVER_ERROR_CANNOT_COMPILE_SENSITIVITY_NOT_COMPUTED = 130505,
    SOLVER_ERROR_GCC_FORK_FAILED = 130506,
    SOLVER_ERROR_DL_LOAD_FAILED = 130507,
    SOLVER_ERROR_DL_SYMBOL_UNDEFINED = 130508,
    SOLVER_ERROR_OPEN_FILE = 130509,
      
    /** 14XXXX - assorted API errors */
    SOLVER_ERROR_SYMBOL_IS_NOT_IN_MODEL = 140000,
    SOLVER_ERROR_ATTEMPTING_TO_COPY_VARIABLE_STATE_BETWEEN_INSTANCES_OF_DIFFERENT_MODELS = 140001,
    SOLVER_ERROR_ATTEMPTING_TO_SET_IMPOSSIBLE_INITIAL_TIME = 140002,
    SOLVER_ERROR_ATTEMPT_TO_SET_ASSIGNED_VALUE = 140003
  } ;

/** error types */
enum errorType
  {
    FATAL_ERROR_TYPE = 0,
    ERROR_ERROR_TYPE = 1,
    WARNING_ERROR_TYPE = 2,
    MESSAGE_ERROR_TYPE = 3,
    NUMBER_OF_ERROR_TYPES = 4
  } ;

typedef enum errorCode errorCode_t;
typedef enum errorType errorType_t;



#define RETURN_ON_ERRORS_WITH(x)				\
  {if (SolverError_getNum(ERROR_ERROR_TYPE) ||			\
       SolverError_getNum(FATAL_ERROR_TYPE)) return (x); }

#define RETURN_ON_FATALS_WITH(x)				\
  {if (SolverError_getNum(FATAL_ERROR_TYPE)) return (x); }

#define ASSIGN_NEW_MEMORY_BLOCK(_ref, _num, _type, _return)	\
  { (_ref) = (_type *)SolverError_calloc(_num, sizeof(_type));	\
    RETURN_ON_FATALS_WITH(_return) }

#define ASSIGN_NEW_MEMORY(_ref, _type, _return)		\
  ASSIGN_NEW_MEMORY_BLOCK(_ref, 1, _type, _return)


#ifdef __cplusplus
extern "C" {
#endif

  /* get number of stored errors  of given type */
  SBML_ODESOLVER_API int SolverError_getNum(errorType_t); 

  /* get a stored error message */
  SBML_ODESOLVER_API char * SolverError_getMessage(errorType_t, int errorNum);

  /* get error code */
  SBML_ODESOLVER_API errorCode_t SolverError_getCode(errorType_t, int errorNum);

  /* get error code of last error stored of given type */
  SBML_ODESOLVER_API errorCode_t SolverError_getLastCode(errorType_t);

  /* empty error store */
  SBML_ODESOLVER_API void SolverError_clear();

  /* create an error */
  SBML_ODESOLVER_API void SolverError_error(errorType_t, errorCode_t, char *format, ...);

#ifdef WIN32
  /* create an error from the last windows error */
  SBML_ODESOLVER_API void SolverError_storeLastWin32Error(const char *context);
#endif

  /* exit the program if errors or fatals have been created. */
  SBML_ODESOLVER_API void SolverError_haltOnErrors();

  /* write all errors and warnings to standard error */
  SBML_ODESOLVER_API void SolverError_dump();

  /* write all errors and warnings to a string (owned by caller unless SolverError_isMemoryExhausted()) */
  SBML_ODESOLVER_API char *SolverError_dumpToString();

  /* free string returned by SolverError_dumpToString */
  SBML_ODESOLVER_API void SolverError_freeDumpString(char *);

  /* write all errors and warnings to standard error and then empty error store*/
  SBML_ODESOLVER_API void SolverError_dumpAndClearErrors();

  /* allocated memory and sets error if it fails */
  SBML_ODESOLVER_API void *SolverError_calloc(size_t num, size_t size);

  /* returns 1 if memory has been exhausted 0 otherwise */
  SBML_ODESOLVER_API int SolverError_isMemoryExhausted();

#ifdef __cplusplus
}
#endif

#endif 
/* _SOLVERERROR_H_ */

