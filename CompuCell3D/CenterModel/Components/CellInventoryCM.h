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

#ifndef CELLINVENTORYCM_H
#define CELLINVENTORYCM_H

#include "ComponentsDLLSpecifier.h"
#include <PublicUtilities/Vector3.h>
#include <map>




namespace CenterModel {


class CellFactoryCM;
class CellCM;

class COMPONENTS_EXPORT CellIdentifierCM{
	public:
		CellIdentifierCM(long _cellId=0 ):cellId(_cellId){}
		long cellId;
		
         ///have to define < operator if using a class in the set and no < operator is defined for this class
         bool operator<(const CellIdentifierCM & _rhs) const{
            return cellId < _rhs.cellId;
         }

};

class COMPONENTS_EXPORT CellInventoryCM{
   public:
	   typedef  std::map<CellIdentifierCM,CellCM *> cellInventoryContainerTypeCM;
	   typedef  cellInventoryContainerTypeCM::iterator cellInventoryIterator;
	   CellInventoryCM():cellFactoryPtr(0)
      {}
      

      virtual ~CellInventoryCM();	  
      virtual void addToInventory(CellCM * _cell);
      virtual void removeFromInventory(CellCM * _cell);

	  void setCellFactory(CellFactoryCM * _cellFactoryPtr){cellFactoryPtr=_cellFactoryPtr;}
	  
      ////std::set<CellG *>::size_type getCellInventorySize(){return inventory.size();}
      int getSize(){return inventory.size();}   
      cellInventoryIterator cellInventoryBegin(){return inventory.begin();}
      cellInventoryIterator cellInventoryEnd(){return inventory.end();}
      void incrementIterator(cellInventoryIterator & _itr){++_itr;}
      void decrementIterator(cellInventoryIterator & _itr){--_itr;}
      CellCM *getCellById(long _id);  



   private:
    
	 cellInventoryContainerTypeCM inventory;
	 CellFactoryCM * cellFactoryPtr;
            
   };



};
#endif
