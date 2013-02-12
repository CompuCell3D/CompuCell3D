/*************************************************************************
 *    CompuCell - A software framework for multimodel simulations of     *
 * biocomplexity problems Copyright (C) 2003 University of Notre Dame,   *
 *                             Indiana                                   *
 *                                                                       *
 * This program is free software; IF YOU AGREE TO CITE USE OF CompuCell  *
 *  IN ALL RELATED RESEARCH PUBLICATIONS according to the terms of the   *
 *  CompuCell GNU General Public License RIDER you can redistribute it   *
 * and/or modify it under the terms of the GNU General Public License as *
 *  published by the Free Software Foundation; either version 2 of the   *
 *         License, or (at your option) any later version.               *
 *                                                                       *
 * This program is distributed in the hope that it will be useful, but   *
 *      WITHOUT ANY WARRANTY; without even the implied warranty of       *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU    *
 *             General Public License for more details.                  *
 *                                                                       *
 *  You should have received a copy of the GNU General Public License    *
 *     along with this program; if not, write to the Free Software       *
 *      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.        *
 *************************************************************************/

 #include <CompuCell3D/CC3D.h>

// // // #include <CompuCell3D/Automaton/Automaton.h>
// // // #include <CompuCell3D/Field3D/Dim3D.h>
// // // #include <CompuCell3D/ClassRegistry.h>
// // // #include <CompuCell3D/Simulator.h>
// // // //#include <CompuCell3D/Diffusable.h>
// // // #include <CompuCell3D/Field3D/Field3D.h>
// // // #include <CompuCell3D/Field3D/Field3DIO.h>
// // // #include <CompuCell3D/Potts3D/Cell.h>
// // // //#include <CompuCell3D/DiffusionSolverBiofilmFE.h>
#include <CompuCell3D/steppables/PDESolvers/DiffusableVector.h>
using namespace CompuCell3D;


// // // #include <BasicUtils/BasicString.h>
// // // #include <BasicUtils/BasicException.h>
// // // #include <PublicUtilities/StringUtils.h>


// // // #include <fstream>
// // // #include <string>
using namespace std;


#include "ChemotaxisSimpleEnergy.h"

float ChemotaxisSimpleEnergy::simpleChemotaxisFormula(float _flipNeighborConc,float _conc,double _lambda){
   return (_flipNeighborConc-_conc)*_lambda;
}




