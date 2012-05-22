/*
  Last changed Time-stamp: <2008-05-09 23:26:31 raim>
  $Id: arithmeticCompiler.h,v 1.6 2009/02/10 12:42:39 stefan_tbi Exp $
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
 *     Matthias Rosensteiner
 *     Markus Loeberbauer
 *
 * Contributor(s):
 *     Stefan Müller
 *     Rainer Machne
 */

#ifndef _ARITHMETICCOMPILER_H_
#define _ARITHMETICCOMPILER_H_

/* #define ARITHMETIC_TEST */ /* comment in test arithmeticCompiler.c */

typedef struct directCode directCode_t;

/*!!!! NOTE: includes need to be below typedef definition to avoid
   circular include problem, but struct declarations needs to
   be after includes, however some compilers require struct before typedef */
#include "sbmlsolver/ASTIndexNameNode.h"
#include "sbmlsolver/cvodeData.h"

struct directCode
{
  int codeSize, FPUstackSize, storageSize;
  int codePosition, FPUstackPosition, storagePosition;
  unsigned char *prog;
  double *FPUstack, *storage;
  double (*evaluate)(cvodeData_t*);
  ASTNode_t *eqn;
  long long *temp;
} ;

void generateFunction(directCode_t *, ASTNode_t *);
void destructFunction(directCode_t *);

#endif /* _ARITHMETICCOMPILER_H_ */
