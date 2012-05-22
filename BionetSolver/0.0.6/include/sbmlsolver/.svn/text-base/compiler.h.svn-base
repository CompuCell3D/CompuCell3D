/*
  Last changed Time-stamp: <2007-08-20 21:06:02 raim>
  $Id: compiler.h,v 1.8 2009/03/27 15:55:03 fbergmann Exp $
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
 *    Christoph Flamm, Rainer Machne
 */

#ifndef _COMPILER_H_
#define _COMPILER_H_

#ifdef WIN32
#include <WINDOWS.h>
#endif

#include "sbmlsolver/exportdefs.h"

#ifdef __cplusplus
extern "C" {
#endif
  
#ifndef WIN32
#include "config.h"
#define MAX_PATH 256
#endif


#if USE_TCC == 1
#include <libtcc.h>
#endif
  
  
  /**
     A structure that stores compiled code
  */
  struct compiled_code
  {

#if USE_TCC == 1
    TCCState *s;
#else
#ifdef WIN32
    HMODULE dllHandle;
#else
    void *dllHandle;
#endif /* WIN32 */
    char *dllFileName;
#endif /* USE_TCC == 1 */

  };

  /* the compiled code structure */
  typedef struct compiled_code compiled_code_t ;

  /**
   * create compiled code from C source
   
   *   On windows this creates a DLL and loads it
   *   On Liunx it will use libtcc and in memory compilation
   *   -- this is better!!
   */
  SBML_ODESOLVER_API compiled_code_t *Compiler_compile(const char *sourceCode);

  /**
   * get pointer to given function corresponding to the symbol
   * in the compiled code
   
   *   On windows use WIN32 API to locate function in dll
   *   On Linux use libtcc to locate function
   */
  SBML_ODESOLVER_API void *CompiledCode_getFunction(compiled_code_t *, const char *symbol);

  /**
   * discard compiled code - don't call this until you have stopped
   * calling the functions returned by getFunction.
   
   *   On windows use Win32 to unlink dll and delete dll
   *   On Linux use libtcc to discard in memory code
   */
  SBML_ODESOLVER_API void CompiledCode_free(compiled_code_t *);

#ifdef __cplusplus
};
#endif

#endif
