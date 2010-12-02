/*
  Last changed Time-stamp: <2008-10-16 16:52:50 raim>
  $Id: sbml.c,v 1.18 2008/10/16 17:27:50 raimc Exp $
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
 *     Rainer Machne
 *
 * Contributor(s):
 *     Christoph Flamm
 *     Andrew M. Finney
 */

#include <stdio.h>
#include <stdlib.h>

/* libSBML header files */
#include <sbmlsolver/SBMLTypesWithoutUnitHeaders.h>
#include <sbml/SBMLDocument.h>


/* own header files */
#include "sbmlsolver/sbml.h"
#include "sbmlsolver/util.h"
#include "sbmlsolver/solverError.h"

void storeSBMLError(errorType_t type, const SBMLError_t *error )
{  
  SolverError_error(type, XMLError_getErrorId((const XMLError_t *)error),
		    "libSBML ERROR\n\t\t\tSEVERITY: %d\n\t\t\tMESSAGE:  %s\n",
		    XMLError_getSeverity((const XMLError_t *)error),
		    XMLError_getMessage((const XMLError_t *)error));
}

/** Loads, validates and parses an SBML file

    also converts SBML level 1 to level 2 files, return NULL if errors
    were encountered during libSBML validation and consistency check,
    stores libSBML warngings */
SBMLDocument_t *
parseModel(char *file, int printMessage, int validate)
{
  unsigned int i, errors, severity;
  SBMLDocument_t *d;
  SBMLDocument_t *d2;
  SBMLReader_t *sr;
  const SBMLError_t *error;

  if (  printMessage )
  {
    fprintf(stderr, "Validating SBML.\n");
    fprintf(stderr, "This can take a while for SBML level 2.\n");
  }
  
  sr = SBMLReader_create();
  d = SBMLReader_readSBML(sr, file);
  SBMLReader_free(sr);

  errors = SBMLDocument_getNumErrors(d); 
    
  /*!!! redo error handling with libsbml 3 */
  if ( errors != 0 && validate )
    errors += SBMLDocument_checkConsistency(d); 

  for ( i =0 ; i != errors; i++ )
  {
    error = SBMLDocument_getError(d, i);
    severity = XMLError_getSeverity((const XMLError_t *)error);
    if ( severity < 2 ) /* infos and warnings */
      storeSBMLError(WARNING_ERROR_TYPE, error);
    else
    {
      /* errors and fatals */
      storeSBMLError(FATAL_ERROR_TYPE, error);
      SBMLDocument_free(d);
      return NULL;
    }
  }


  
  /* convert level 1 models to level 2 */
  if ( SBMLDocument_getLevel(d) == 1 )
  {
    d2 = convertModel(d); 
    SBMLDocument_free(d);
 
    if ( printMessage )
      fprintf(stderr, "SBML converted from level 1 to level 2.\n"); 
    d = d2; 
  } 
 
  return (d);
}

SBMLDocument_t *convertModel (SBMLDocument_t *d1)
{
  int i, severity;
  SBMLDocument_t *d2;
  const SBMLError_t *error;
  
  d2 = SBMLDocument_clone(d1);
  SBMLDocument_setLevelAndVersion(d2, 2, 1);
  
  for ( i =0 ; i != SBMLDocument_getNumErrors(d1); i++ )
  {
    error = SBMLDocument_getError(d1, i);
    severity = XMLError_getSeverity((const XMLError_t *)error);
    if ( severity < 2 ) /* infos and warnings */
      storeSBMLError(WARNING_ERROR_TYPE, error);
    else
    {
      /* errors and fatals */
      storeSBMLError(FATAL_ERROR_TYPE, error);
      SBMLDocument_free(d2);
      return NULL;
    }
  }  

  return d2;
}

/* End of file */
