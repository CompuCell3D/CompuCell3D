/*
  Last changed Time-stamp: <2008-09-24 12:18:17 raim>
  $Id: processAST.h,v 1.24 2009/03/27 15:55:03 fbergmann Exp $
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
 *     Stefan Müller
 */

#ifndef _PROCESSAST_H_
#define _PROCESSAST_H_

/* libSBML header files */
#include <sbmlsolver/SBMLTypesWithoutUnitHeaders.h>

/* own header files */
#include "sbmlsolver/odeModel.h"
#include "sbmlsolver/cvodeData.h"
#include "sbmlsolver/exportdefs.h"
#include "sbmlsolver/charBuffer.h"

#define MySQR(x) ((x)*(x))
#define MySQRT(x) pow((x),(.5))
/* Helper Macros to get the second or the third child
   of an Abstract Syntax Tree */
#define child(x,y)  ASTNode_getChild(x,y)
#define child2(x,y,z)  ASTNode_getChild(ASTNode_getChild(x,y),z)
#define child3(x,y,z,w) ASTNode_getChild(ASTNode_getChild(ASTNode_getChild(x,y),z),w)

#ifdef __cplusplus
extern "C" {
#endif
  
  SBML_ODESOLVER_API double evaluateAST(ASTNode_t *n, cvodeData_t *data);
  SBML_ODESOLVER_API void generateMacros(charBuffer_t *buffer);
  SBML_ODESOLVER_API void generateAST(charBuffer_t *buffer, const ASTNode_t *n);
  SBML_ODESOLVER_API ASTNode_t *differentiateAST(ASTNode_t *f, char*x);
  SBML_ODESOLVER_API ASTNode_t *AST_simplify(ASTNode_t *f);
  SBML_ODESOLVER_API void setUserDefinedFunction(double(*udf)(char*, int, double*));
  SBML_ODESOLVER_API ASTNode_t *copyAST(const ASTNode_t *f);
  SBML_ODESOLVER_API ASTNode_t *determinantNAST(ASTNode_t ***A, int N);
  SBML_ODESOLVER_API ASTNode_t *simplifyAST(ASTNode_t *f);
  SBML_ODESOLVER_API ASTNode_t *ASTNode_indexAST(const ASTNode_t *f, odeModel_t *om);
  
  ASTNode_t *indexAST(const ASTNode_t *f, int nvalues, char ** names);
  void AST_dump(const char *context, ASTNode_t *node);
  void ASTNode_getSymbols(ASTNode_t *node, List_t *symbols);
  int ASTNode_containsTime(ASTNode_t *node);
  int ASTNode_containsPiecewise(ASTNode_t *node);
  int ASTNode_getIndices(ASTNode_t *node, List_t *indices);
  int *ASTNode_getIndexArray(ASTNode_t *node, int nvalues);


#ifdef __cplusplus
};
#endif

#endif

/* End of file */
