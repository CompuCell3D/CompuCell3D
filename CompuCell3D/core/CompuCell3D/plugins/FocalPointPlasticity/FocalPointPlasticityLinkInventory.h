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
#ifndef FOCALPOINTPLASTICITYLINKINVENTORY_H
#define FOCALPOINTPLASTICITYLINKINVENTORY_H

#include "FocalPointPlasticityLinkInventoryBase.h"

namespace CompuCell3D {

	/**
	Written by T.J. Sego, Ph.D.
	*/

	class FocalPointPlasticityLink;
	class FocalPointPlasticityInternalLink;
	class FocalPointPlasticityAnchor;

	class FOCALPOINTPLASTICITY_EXPORT FPPLinkInventory : public FPPLinkInventoryBase<FocalPointPlasticityLink> {

	public:

		FPPLinkInventory() {}
		FPPLinkInventory(BasicClassAccessor<FocalPointPlasticityTracker>* _focalPointPlasticityTrackerAccessor)
		{
			focalPointPlasticityTrackerAccessor = _focalPointPlasticityTrackerAccessor;
		}
		virtual ~FPPLinkInventory() {}

		FPPTrackerDataSet& getFPPTrackerDataSet(CellG* _cell) {
			if (focalPointPlasticityTrackerAccessor)
				return focalPointPlasticityTrackerAccessor->get(_cell->extraAttribPtr)->focalPointPlasticityNeighbors;
			else return FPPTrackerDataSet();
		}
		// Get the link connecting two cells
		FocalPointPlasticityLink* getLinkByCells(CellG* _cell0, CellG* _cell1) { return getLinkById(FPPLinkID(_cell0->id, _cell1->id)); }
		// Get list of cells linked to a cell
		const std::vector<const CellG*> getLinkedCells(CellG* _cell) {
			std::vector<const CellG*> o;
			FPPInventory_t *cInv = getCellLinkInventory(_cell);
			for (linkInventoryItr_t itr = cInv->linkInventoryBegin(); itr != cInv->linkInventoryEnd(); ++itr) {
				o.push_back(itr->second->getOtherCell(_cell));
			}
			return o;
		}
		// Get number of junctions for a cell by type
		const int getNumberOfJunctionsByType(CellG* _cell, long _type) {
			FPPInventory_t *cInv = getCellLinkInventory(_cell);
			return count_if(cInv->linkInventoryBegin(), cInv->linkInventoryEnd(), [&](linkInventoryPair_t p) {
				return p.second->getOtherCell(_cell)->type == _type;
			});
		}

	};

	class FOCALPOINTPLASTICITY_EXPORT FPPInternalLinkInventory : public FPPLinkInventoryBase<FocalPointPlasticityInternalLink> {

	public:

		FPPInternalLinkInventory() {}
		FPPInternalLinkInventory(BasicClassAccessor<FocalPointPlasticityTracker>* _focalPointPlasticityTrackerAccessor)
		{
			focalPointPlasticityTrackerAccessor = _focalPointPlasticityTrackerAccessor;
		}
		virtual ~FPPInternalLinkInventory() {}

		FPPTrackerDataSet& getFPPTrackerDataSet(CellG* _cell) {
			if (focalPointPlasticityTrackerAccessor)
				return focalPointPlasticityTrackerAccessor->get(_cell->extraAttribPtr)->internalFocalPointPlasticityNeighbors;
			else return FPPTrackerDataSet();
		}
		// Get the link connecting two cells
		FocalPointPlasticityInternalLink* getLinkByCells(CellG* _cell0, CellG* _cell1) { return getLinkById(FPPLinkID(_cell0->id, _cell1->id)); }
		// Get list of cells linked to a cell
		const std::vector<const CellG*> getLinkedCells(CellG* _cell) {
			std::vector<const CellG*> o;
			FPPInventory_t *cInv = getCellLinkInventory(_cell);
			for (linkInventoryItr_t itr = cInv->linkInventoryBegin(); itr != cInv->linkInventoryEnd(); ++itr) {
				o.push_back(itr->second->getOtherCell(_cell));
			}
			return o;
		}
		// Get number of junctions for a cell by type
		const int getNumberOfJunctionsByType(CellG* _cell, long _type) {
			FPPInventory_t *cInv = getCellLinkInventory(_cell);
			return count_if(cInv->linkInventoryBegin(), cInv->linkInventoryEnd(), [&](linkInventoryPair_t p) {
				return p.second->getOtherCell(_cell)->type == _type;
			});
		}

	};
	
	class FOCALPOINTPLASTICITY_EXPORT FPPAnchorInventory : public FPPLinkInventoryBase<FocalPointPlasticityAnchor> {

	public:

		FPPAnchorInventory() {}
		FPPAnchorInventory(BasicClassAccessor<FocalPointPlasticityTracker>* _focalPointPlasticityTrackerAccessor)
		{
			focalPointPlasticityTrackerAccessor = _focalPointPlasticityTrackerAccessor;
		}
		virtual ~FPPAnchorInventory() {}

		FPPTrackerDataSet& getFPPTrackerDataSet(CellG* _cell) {
			if (focalPointPlasticityTrackerAccessor)
				return focalPointPlasticityTrackerAccessor->get(_cell->extraAttribPtr)->anchors;
			else return FPPTrackerDataSet();
		}

		FocalPointPlasticityAnchor* getAnchor(CellG* _cell, long _anchorId) { return getLinkById(FPPLinkID(_cell->id, _anchorId)); }

		int getNextAnchorId(CellG* _cell) {
			FPPLinkList ll = getCellLinkList(_cell);
			if (ll.size() == 0) return int(0);
			else return (*ll.end())->getAnchorId() + 1;
		}

	};

}

#endif