/*
  Last changed Time-stamp: <2005-10-26 17:24:50 raim>
  $Id: exportdefs.h,v 1.5 2005/10/26 15:32:13 raimc Exp $
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
 */

/* The following ifdef block is the standard way of creating macros
which make exporting from a DLL simpler. All files within this DLL are
compiled with the SBML_ODESOLVER_EXPORTS symbol defined on the command
line. this symbol should not be defined on any project that uses this
DLL. This way any other project whose source files include this file
see SBML_ODESOLVER_API functions as being imported from a DLL, whereas
this DLL sees symbols defined with this macro as being exported. */
#ifdef WIN32
#ifdef SBML_ODESOLVER_EXPORTS
#define SBML_ODESOLVER_API __declspec(dllexport)
#else
#define SBML_ODESOLVER_API __declspec(dllimport)
#endif
#else
#define SBML_ODESOLVER_API
#endif

/* examples of use...

// This class is exported from the SBML_odeSolver.dll
class SBML_ODESOLVER_API CSBML_odeSolver {
public:
	CSBML_odeSolver(void);
	// TODO: add your methods here.
};

extern SBML_ODESOLVER_API int nSBML_odeSolver;

SBML_ODESOLVER_API int fnSBML_odeSolver(void);

*/

