/*
  Last changed Time-stamp: <2008-10-16 17:55:16 raim>
  $Id: compiler.c,v 1.32 2009/03/27 15:54:55 fbergmann Exp $
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

#include <stdio.h>
#include "compiler.h"
#include "solverError.h"
#include "processAST.h"

#ifdef WIN32

#include <windows.h>
#include <tchar.h>

#else /* default linux */

#include "sbmlsolver/config.h"
#include "sbmlsolver/util.h"
#include <dlfcn.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

#endif /* end WIN32 */


#ifdef WIN32

compiled_code_t *Compiler_compile_win32_tcc(const char *sourceCode)
{
  compiled_code_t *code = NULL;
  char tempDir[MAX_PATH+1];
  TCHAR tccFileName[MAX_PATH]; 
  int i;
  int result;
  char *cFileName;
  char *dllFileName;
  char *outFileName;
  FILE *cFile;
  char command[4*MAX_PATH];
  char *dllFileNameDot ;
  HMODULE dllHandle, solverHandle ;
#ifdef _DEBUG
  char *solverFileName = "SBML_odeSolverD.dll";
#else
  char *solverFileName = "SBML_odeSolver.dll";
#endif

  /*printf("Source code:\n%s\n", sourceCode);*/

  /* avoid creating files in current directory if environment
     variables not set */
  if (!GetTempPath(MAX_PATH+1, tempDir))
  {
    SolverError_storeLastWin32Error("Trying to find out location of system temp directory");
    return NULL;
  }
  
  solverHandle = GetModuleHandle(solverFileName);
    
  if (!solverHandle)
  {
    SolverError_storeLastWin32Error("Trying to get handle of solver dll");
    return NULL;
  }
  
  /* compute tcc path from the path to this dll */
  if( !GetModuleFileName( solverHandle, tccFileName, MAX_PATH ) )
  {
    SolverError_storeLastWin32Error("Trying find location of the soslib dll");
    return NULL ;
  }

  for (i = strlen(tccFileName); i != -1 && tccFileName[i] != '\\'; i--);

  tccFileName[i + 1] = '\0';
  strcat(tccFileName, "tcc\\tcc.exe");

  cFileName = tempnam(tempDir, "temp_soslib_c_file");
  dllFileName = tempnam(tempDir, "temp_soslib_dll");
  outFileName = tempnam(tempDir, "temp_soslib_compilation_output");
  cFile = fopen(cFileName, "w");

  if (!cFile)
  {
    SolverError_storeLastWin32Error("Unable to open C source file for write");
    return NULL;
  }

  fprintf(cFile, "%s", sourceCode);
  fclose(cFile);

  sprintf(command, "%s -o %s -shared %s > %s",
	  tccFileName, dllFileName, cFileName, outFileName);

  /*printf("Command:\n%s\n", command);*/

  result = system(command);

  if (result == -1)
  {
    SolverError_storeLastWin32Error("Whilst running compile command");
    remove(cFileName);
    free(cFileName);
    return NULL ;
  }
  else if (result != 0)
  {
    SolverError_error(ERROR_ERROR_TYPE, SOLVER_ERROR_COMPILATION_FAILED,
		      "Compile command failed - returned %d", result);
    remove(cFileName);
    free(cFileName);
    return NULL ;
  }
  
  remove(cFileName);
  free(cFileName);
  remove(outFileName);
  free(outFileName);

  ASSIGN_NEW_MEMORY_BLOCK(dllFileNameDot, (strlen(dllFileName) +2), char, NULL);
  
  strcpy(dllFileNameDot, dllFileName);
  strcat(dllFileNameDot, ".");

  dllHandle = LoadLibrary(dllFileNameDot);
  free(dllFileNameDot);

  if (!dllHandle)
  {
    SolverError_storeLastWin32Error("While loading compiled dll");
    return NULL;
  }
  
  ASSIGN_NEW_MEMORY(code, compiled_code_t, NULL);

  code->dllHandle = dllHandle ;
  code->dllFileName = dllFileName;

  return (code);
}

#elif defined(_AIX) || defined(__AIX) || defined(__AIX__) || defined(__aix) || defined(__aix__) /* AIX use xlc_r */

/**
   Returns a pointer to code that is compiled from the given source code
*/
compiled_code_t *Compiler_compile_with_xlc(const char *sourceCode)
{
  compiled_code_t *code = NULL;
  char gccFileName[MAX_PATH+1] = "xlc_r"; 
  int result;
  char *tmpFileName = NULL;
  char *cFileName   = NULL;
  char *dllFileName = NULL;
  char *oFileName   = NULL;
  FILE *cFile;
  char command[4*MAX_PATH];
  void *dllHandle;
  
  /* generate a unique temprorary filename template */
  ASSIGN_NEW_MEMORY_BLOCK(tmpFileName, (MAX_PATH+1), char, NULL);
  tmpFileName = tmpnam(tmpFileName);
  
#ifdef _DEBUG
  Warn(NULL,"Temporary File Name is %s\n", tmpFileName);
#endif
  
  /* generate needed file names from the template*/
  ASSIGN_NEW_MEMORY_BLOCK(cFileName, (strlen(tmpFileName)+3), char, NULL);
  strcpy(cFileName, tmpFileName);
  strcat(cFileName, ".c");
  ASSIGN_NEW_MEMORY_BLOCK(oFileName, (strlen(tmpFileName)+3), char, NULL);
  strcpy(oFileName, tmpFileName);
  strcat(oFileName, ".o");  
  ASSIGN_NEW_MEMORY_BLOCK(dllFileName,
			  (strlen(tmpFileName)+strlen(SHAREDLIBEXT)+1),
			  char, NULL);
  strcpy(dllFileName, tmpFileName);
  strcat(dllFileName, SHAREDLIBEXT);
  
  /* open file and dump source code to it */
  cFile = fopen(cFileName, "w");
  
  if (!cFile)
  {
    SolverError_error(WARNING_ERROR_TYPE, SOLVER_ERROR_OPEN_FILE,
		      "Could not open file %s - %s!",
		      cFileName, strerror(errno));
    free(cFileName);
    free(oFileName);
    free(dllFileName);
    return NULL;
  }
  
  fprintf(cFile, "%s", sourceCode);
  fclose(cFile);
  
  /* construct command for compiling */
  sprintf(command, "%s -I%s -I%s -I%s -I../src -G -o %s %s -L../src -L%s -L%s -L%s -lODES -lsbml -lm",
	  gccFileName,
	  SOSLIB_CFLAGS, /* changed order: SOSLIB first */
	  SUNDIALS_CFLAGS,
	  SBML_CFLAGS,
	  dllFileName,
	  cFileName,
	  SUNDIALS_LDFLAGS,
	  SBML_LDFLAGS,
	  SOSLIB_LDFLAGS);
  
#ifdef _DEBUG
  Warn(NULL, "Command: %s\n", command);
#endif
  
  /* compile source to shared library */
  result = system(command);
  
  /* clean up compilation intermediates */
  free(tmpFileName);
  remove(cFileName);
  free(cFileName);
  remove(oFileName);
  free(oFileName);

  /* handle possible errors */
  if (result != 0)
  {
    if (result == -1)
      SolverError_error(WARNING_ERROR_TYPE, SOLVER_ERROR_GCC_FORK_FAILED,
			"forking xlc compiler subprocess failed!");
    else 
      SolverError_error(WARNING_ERROR_TYPE, SOLVER_ERROR_COMPILATION_FAILED,
			"compiling failed with errno %d - %s!",
			result, strerror(result));    
    return (NULL);
  }
  
  
  /* load shared library */
  dllHandle = dlopen(dllFileName, RTLD_LAZY);
  if (dllHandle == NULL)
  {
    SolverError_error(WARNING_ERROR_TYPE, SOLVER_ERROR_DL_LOAD_FAILED,
		      "loading shared library %s failed %d - %s!",
		      dllFileName, errno, strerror(errno));
    /* SolverError_dumpAndClearErrors(); */
    return (NULL);
  }
  
  ASSIGN_NEW_MEMORY(code, compiled_code_t, NULL);
  code->dllHandle   = dllHandle;
  code->dllFileName = dllFileName;
  
  return (code);
}

#else /* default case is compile with gcc */

/**
   Returns a pointer to code that is compiled from the given source code
*/
compiled_code_t *Compiler_compile_with_gcc(const char *sourceCode)
{
  compiled_code_t *code = NULL;
  char gccFileName[MAX_PATH+1] = "gcc"; 
  int result;
  char *tmpFileName = NULL;
  char *cFileName   = NULL;
  char *dllFileName = NULL;
  char *oFileName   = NULL;
  FILE *cFile;
  char command[4*MAX_PATH];
  void *dllHandle;
  
  /* generate a unique temprorary filename template */
  ASSIGN_NEW_MEMORY_BLOCK(tmpFileName, (MAX_PATH+1), char, NULL);
  tmpFileName = tmpnam(tmpFileName);
  
#ifdef _DEBUG
  Warn(NULL,"Temporary File Name is %s\n", tmpFileName);
#endif
  
  /* generate needed file names from the template*/
  ASSIGN_NEW_MEMORY_BLOCK(cFileName, (strlen(tmpFileName)+3), char, NULL);
  strcpy(cFileName, tmpFileName);
  strcat(cFileName, ".c");
  ASSIGN_NEW_MEMORY_BLOCK(oFileName, (strlen(tmpFileName)+3), char, NULL);
  strcpy(oFileName, tmpFileName);
  strcat(oFileName, ".o");  
  ASSIGN_NEW_MEMORY_BLOCK(dllFileName,
			  (strlen(tmpFileName)+strlen(SHAREDLIBEXT)+1),
			  char, NULL);
  strcpy(dllFileName, tmpFileName);
  strcat(dllFileName, SHAREDLIBEXT);

  /* open file and dump source code to it */
  cFile = fopen(cFileName, "w");
  
  if (!cFile)
  {
    SolverError_error(WARNING_ERROR_TYPE, SOLVER_ERROR_OPEN_FILE,
		      "Could not open file %s - %s!",
		      cFileName, strerror(errno));  
    free(cFileName);
    free(oFileName);
    free(dllFileName);
    return NULL;
  }

  fprintf(cFile, "%s", sourceCode);
  fclose(cFile);

  /* construct command for compiling */
#if defined (__APPLE__) && defined (__MACH__)
  sprintf(command,
	  "%s -I%s -I%s -I%s -I../src -pipe -O -dynamiclib -fPIC -o %s %s -L../src -L%s -L%s -L%s -lODES -lsbml -lm -lstdc++",
 	  gccFileName,
	  SOSLIB_CFLAGS, /* changed order: SOSLIB first */
	  SUNDIALS_CFLAGS,
	  SBML_CFLAGS,
	  dllFileName,
	  cFileName,
	  SUNDIALS_LDFLAGS,
	  SBML_LDFLAGS,
	  SOSLIB_LDFLAGS);
#else
  sprintf(command,
	  "%s -I%s -I%s -I%s -I../src -pipe -O -shared -fPIC -o %s %s -L../src -L%s -L%s -L%s -lODES -lsbml -lm -lstdc++",
 	  gccFileName,
	  SOSLIB_CFLAGS, /* changed order: SOSLIB first */
	  SUNDIALS_CFLAGS,
	  SBML_CFLAGS,
	  dllFileName,
	  cFileName,
	  SUNDIALS_LDFLAGS,
	  SBML_LDFLAGS,
	  SOSLIB_LDFLAGS);
#endif
 
#ifdef _DEBUG
  Warn(NULL, "Command: %s\n", command);
#endif
  
  /* compile source to shared library */
  result = system(command);

  /* clean up compilation intermediates */
  free(tmpFileName);
  remove(cFileName);
  free(cFileName);
  remove(oFileName);
  free(oFileName);

  /* handle possible errors */
  if (result != 0)
  {
    if (result == -1)
      SolverError_error(WARNING_ERROR_TYPE, SOLVER_ERROR_GCC_FORK_FAILED,
			"forking gcc compiler subprocess failed!");
    else 
      SolverError_error(WARNING_ERROR_TYPE, SOLVER_ERROR_COMPILATION_FAILED,
			"compiling failed with errno %d - %s!",
			result, strerror(result));    
    return (NULL);
  }
  


  /* load shared library */
  dllHandle = dlopen(dllFileName, RTLD_LAZY);
  if (dllHandle == NULL)
  {
    SolverError_error(WARNING_ERROR_TYPE, SOLVER_ERROR_DL_LOAD_FAILED,
		      "loading shared library %s failed %d - %s!",
		      dllFileName, errno, strerror(errno));
    SolverError_dumpAndClearErrors();
    return (NULL);
  }

  ASSIGN_NEW_MEMORY(code, compiled_code_t, NULL);
  code->dllHandle   = dllHandle;
  code->dllFileName = dllFileName;

  return (code);
}

#endif /* end WIN32 */

/**
   Returns a pointer to code that is compiled from the given source code
*/
compiled_code_t *Compiler_compile(const char *sourceCode)
{
  compiled_code_t *code = NULL;

#ifdef WIN32

  //code = Compiler_compile_with_tcc(sourceCode);

#elif defined(_AIX) || defined(__AIX) || defined(__AIX__) || defined(__aix) || defined(__aix__) /* AIX use xlc_r */

  code = Compiler_compile_with_xlc(sourceCode);
  
#else

  code = Compiler_compile_with_gcc(sourceCode);

#endif /* end WIN32 */

  return (code);
}

/**
   returns a pointer to the function named 'symbol' in the given 'code'
*/
void *CompiledCode_getFunction(compiled_code_t *code, const char *symbol)
{
  void *result = NULL;

#ifdef WIN32
  
  result = GetProcAddress(code->dllHandle, symbol);

  if (result)
    return result ;

  SolverError_storeLastWin32Error("");
  result = NULL;

#else /* default case gcc */

  char* returnvalue = NULL;

  /* Clear any existing error */
  returnvalue = dlerror();
  result = dlsym(code->dllHandle, symbol);

  returnvalue = dlerror();
  if ( returnvalue != NULL )
    SolverError_error(FATAL_ERROR_TYPE, SOLVER_ERROR_DL_SYMBOL_UNDEFINED,
		      "dlsym(): couldn't get symbol %s from shared library %s",
		      symbol, code->dllFileName);    

#endif /* end WIN32 */
  
  return (result);
}

/**
   frees the given code
*/
void CompiledCode_free(compiled_code_t *code)
{

#ifdef WIN32

  FreeLibrary(code->dllHandle);
  remove(code->dllFileName);
  free(code->dllFileName);
  free(code);

#else /* default case gcc */

  dlclose(code->dllHandle);
  remove(code->dllFileName);
  free(code->dllFileName);
  free(code);

#endif /* end WIN32 */

}
