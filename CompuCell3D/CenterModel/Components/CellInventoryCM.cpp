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

#include "CellInventoryCM.h"
#include "CellFactoryCM.h"

#include "CellCM.h"

using namespace std;
using namespace CenterModel;



      

CellInventoryCM::~CellInventoryCM(){

		//Freeing up cell inventory has to be done
		CellInventoryCM::cellInventoryIterator cInvItr;

		CellCM * cell;
		

		///loop over all the cells in the inventory   
		for( cInvItr = cellInventoryBegin() ; cInvItr !=cellInventoryEnd() ; ++cInvItr ){
			cell=cInvItr->second;			
			if(!cellFactoryPtr){
				delete cell;
         }
         else{
				cellFactoryPtr->destroyCellCM(cell);
         }
		}
	inventory.clear();	

}


void CellInventoryCM::addToInventory(CellCM * _cell){
	inventory.insert(make_pair(CellIdentifierCM(_cell->id),_cell ));
}

void CellInventoryCM::removeFromInventory(CellCM * _cell){
	inventory.erase(CellIdentifierCM(_cell->id));
}
	  
CellCM * CellInventoryCM::getCellById(long _id){
    cellInventoryContainerTypeCM::iterator mitr=inventory.find(CellIdentifierCM(_id));
    if (mitr!=inventory.end()){
        return mitr->second;
    }
    return 0;

}