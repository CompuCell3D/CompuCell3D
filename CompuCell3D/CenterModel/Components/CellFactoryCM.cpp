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

#include "CellFactoryCM.h"
#include "SimulationBox.h"
#include "CellCM.h"

#include <iostream>

using namespace std;
using namespace CenterModel;

CellCM * CellFactoryCM::createCellCM(double _x,double _y, double _z){

    
    CellCM *cell=new CellCM();

	cell->position.fX=_x;
	cell->position.fY=_y;
	cell->position.fZ=_z;

	++recentCellId;
	cell->id=recentCellId;

	sbPtr->updateCellLookup(cell); //storin cell in a lookup set
	return cell;
}

bool CellFactoryCM::destroyCellCM(CellCM * _cell){
	delete _cell;
	return true;

}