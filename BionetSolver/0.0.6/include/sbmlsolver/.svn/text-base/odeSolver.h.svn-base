/*
  Last changed Time-stamp: <2008-03-10 20:07:56 raim>
  $Id: odeSolver.h,v 1.27 2009/03/27 15:55:03 fbergmann Exp $
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
 *     
 */

#ifndef _ODESOLVER_H_
#define _ODESOLVER_H_

typedef struct varySettings varySettings_t;

/* libSBML header files */
#include <sbmlsolver/SBMLTypesWithoutUnitHeaders.h>

/* own header files */
#include "sbmlsolver/util.h"
#include "sbmlsolver/sbml.h"
#include "sbmlsolver/modelSimplify.h"
#include "sbmlsolver/cvodeData.h"
#include "sbmlsolver/odeConstruct.h"
#include "sbmlsolver/integratorSettings.h"
#include "sbmlsolver/integratorInstance.h"
#include "sbmlsolver/solverError.h"
#include "sbmlsolver/drawGraph.h"
#include "sbmlsolver/processAST.h"
#include "sbmlsolver/sbmlResults.h"
#include "sbmlsolver/exportdefs.h"

/* needed by sm */
#include "sbmlsolver/ASTIndexNameNode.h"
#include "sbmlsolver/variableIndex.h"
#include "sbmlsolver/odeModel.h"


  /** Settings for batch integration with parameter variation */
  struct varySettings {
    int nrdesignpoints; /**< defines how many design points are set*/
    int nrparams;       /**< defines the number of parameters to be varied */
    char **id;          /**< array of SBML ID of the species, compartment
			     or parameter to be varied */
    char **rid;         /**< SBML Reaction ID, if a local parameter is to be
			     varied */
    double **params;    /**< two dimensional array for parameter values */

    /* just used during construction */
    int cnt_params;     /**< counts the number of parameters added */
    int cnt_points;     /**< counts the number of designpoints filled */
  };



#ifdef __cplusplus
extern "C" {
#endif

  SBML_ODESOLVER_API SBMLResults_t *SBML_odeSolver(SBMLDocument_t *, cvodeSettings_t *);
  SBML_ODESOLVER_API SBMLResultsArray_t *SBML_odeSolverBatch(SBMLDocument_t *, cvodeSettings_t *, varySettings_t *);
  SBML_ODESOLVER_API SBMLResults_t *Model_odeSolver(Model_t *, cvodeSettings_t *);
  SBML_ODESOLVER_API SBMLResultsArray_t *Model_odeSolverBatch(Model_t *, cvodeSettings_t *, varySettings_t *);
  SBML_ODESOLVER_API SBMLResults_t *SBMLResults_fromIntegrator(Model_t *, integratorInstance_t *);

  /* settings for parameter variation batch runs */
  SBML_ODESOLVER_API varySettings_t *VarySettings_allocate(int nrparams, int nrdesignpoints);
  SBML_ODESOLVER_API int VarySettings_addDesignPoint(varySettings_t *, double *);
  SBML_ODESOLVER_API int VarySettings_addParameter(varySettings_t *, char *, char *);
  SBML_ODESOLVER_API int VarySettings_setName(varySettings_t *, int, char *, char *);
  SBML_ODESOLVER_API int VarySettings_setValue(varySettings_t *, int, int, double);
  SBML_ODESOLVER_API double VarySettings_getValue(varySettings_t *, int, int);
  SBML_ODESOLVER_API int VarySettings_setValueByID(varySettings_t *, int, char *, char*, double);
  SBML_ODESOLVER_API double VarySettings_getValueID(varySettings_t *, int, char *, char*);
  SBML_ODESOLVER_API const char *VarySettings_getName(varySettings_t *, int);
  SBML_ODESOLVER_API const char *VarySettings_getReactionName(varySettings_t *, int);

  SBML_ODESOLVER_API void VarySettings_dump(varySettings_t *);
  SBML_ODESOLVER_API void VarySettings_free();


 
#ifdef __cplusplus
};
#endif

#endif

/* End of file */
