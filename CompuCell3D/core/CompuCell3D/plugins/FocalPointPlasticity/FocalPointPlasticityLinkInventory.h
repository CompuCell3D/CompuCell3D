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

#include <map>
#include <set>
#include <vector>

#include "FocalPointPlasticityLinks.h"
#include "FocalPointPlasticityDLLSpecifier.h"

namespace CompuCell3D {

	/**
	Written by T.J. Sego, Ph.D.
	*/

	class FocalPointPlasticityLinkBase;
	class FocalPointPlasticityLink;
	class FocalPointPlasticityInternalLink;
	class FocalPointPlasticityAnchor;

	template <typename T>
	class FPPLinkListBase : public std::vector<T*> {
	public:
		typedef typename std::vector<T*>::iterator FPPLinkListIterator_t;
		virtual ~FPPLinkListBase() {}
	};

	class FOCALPOINTPLASTICITY_EXPORT FPPLinkID {
	public:
		long id0;
		long id1;

		FPPLinkID(long _id0, long _id1)
		{
			if (_id0 < _id1) {
				id0 = _id0;
				id1 = _id1;
			}
			else {
				id1 = _id0;
				id0 = _id1;
			}
		}
		virtual ~FPPLinkID() {}
		bool operator < (const FPPLinkID & _rhs) const { return id0 < _rhs.id0 || (id0 == _rhs.id0 && id1 < _rhs.id1); }
		bool operator == (const FPPLinkID & _rhs) const { return id0 == _rhs.id0 && id1 == _rhs.id1; }
		bool operator != (const FPPLinkID & _rhs) const { return !(operator==(_rhs)); }
	};

	template <class LinkType>
	class FOCALPOINTPLASTICITY_EXPORT FPPLinkInventoryBase {

	public:

		typedef FPPLinkListBase<LinkType> FPPLinkList;

		typedef std::map<const FPPLinkID, LinkType*> linkInventory_t;
		typedef typename linkInventory_t::iterator linkInventoryItr_t;
		typedef std::pair<const FPPLinkID, LinkType*> linkInventoryPair_t;

		typedef std::map<const CellG*, FPPLinkInventoryBase<LinkType> > cellLinkInventory_t;
		typedef typename cellLinkInventory_t::iterator cellLinkInventoryItr_t;
		typedef std::pair<const CellG*, FPPLinkInventoryBase<LinkType> > cellLinkInventoryPair_t;

	protected:

		linkInventory_t linkInventory;

		LinkType* getLinkById(const FPPLinkID _id) {
			linkInventoryItr_t itr = linkInventory.find(_id);
			if (itr != linkInventory.end()) return itr->second;
			return (LinkType*)(0);
		}
		const FPPLinkID getLinkId(LinkType* _link) { return FPPLinkID(_link->getId0(), _link->getId1()); }

		// Add a link without any additional internal work
		void addLinkNoChain(LinkType* _link) {
			linkInventory.insert(linkInventoryPair_t(getLinkId(_link), _link));
		}
		// Remove a link without any additional internal work
		void removeLinkNoChain(LinkType* _link) {
			const FPPLinkID linkId = getLinkId(_link);
			linkInventoryItr_t itr = linkInventory.find(linkId);
			if (itr != linkInventory.end()) linkInventory.erase(linkId);
		}

		cellLinkInventory_t cellLinkInventory;

		FPPLinkInventoryBase<LinkType>* getCellLinkInventory(const CellG* _cell) {
			return getCellLinkInventory(const_cast<CellG*>(_cell));
		}
		FPPLinkInventoryBase<LinkType>* getCellLinkInventory(CellG* _cell) { return &cellLinkInventory[_cell]; }

	public:
		FPPLinkInventoryBase() {

			linkInventory = linkInventory_t();

		};
		virtual ~FPPLinkInventoryBase() {}

		int getLinkInventorySize() { return linkInventory.size(); }
		linkInventoryItr_t linkInventoryBegin() { return linkInventory.begin(); }
		linkInventoryItr_t linkInventoryEnd() { return linkInventory.end(); }
		void incrementIterator(linkInventoryItr_t& _itr) { ++_itr; }
		void decrementIterator(linkInventoryItr_t& _itr) { --_itr; }

		// Add a link to the link inventory and update internals
		void addToInventory(LinkType* _link) {
			addLinkNoChain(_link);
			getCellLinkInventory(_link->getObj0())->addLinkNoChain(_link);
			getCellLinkInventory(_link->getObj1())->addLinkNoChain(_link);
		}
		// Remove a link to the link inventory and update internals
		void removeFromInventory(LinkType* _link) {
			removeLinkNoChain(_link);
			getCellLinkInventory(_link->getObj0())->removeLinkNoChain(_link);
			getCellLinkInventory(_link->getObj1())->removeLinkNoChain(_link);
		}

		// Get link inventory list
		const FPPLinkList getLinkList() {
			FPPLinkList fppLinkList;
			for (linkInventoryItr_t itr = linkInventory.begin(); itr != linkInventory.end(); ++itr)
				fppLinkList.push_back(itr->second);
			return fppLinkList;
		}
		// Get link inventory list by cell
		const FPPLinkList getCellLinkList(CellG* _cell) { return getCellLinkInventory(_cell)->getLinkList(); }
		// Remove all links attached to a cell
		void removeCellLinks(CellG* _cell) {
			FPPLinkList cellLinkList = getCellLinkList(_cell);
			for (FPPLinkList::iterator itr = cellLinkList.begin(); itr != cellLinkList.end(); ++itr)
				removeFromInventory((*itr));
			
			cellLinkInventoryItr_t cItr = cellLinkInventory.find(_cell);
			if (cItr != cellLinkInventory.end()) cellLinkInventory.erase(_cell);
		}

	};

	class FOCALPOINTPLASTICITY_EXPORT FPPLinkInventory : public FPPLinkInventoryBase<FocalPointPlasticityLink> {

	public:

		FPPLinkInventory() {}
		virtual ~FPPLinkInventory() {}

		FocalPointPlasticityLink* getLinkByCells(CellG* _cell0, CellG* _cell1) { return getLinkById(FPPLinkID(_cell0->id, _cell1->id)); }

	};

	class FOCALPOINTPLASTICITY_EXPORT FPPInternalLinkInventory : public FPPLinkInventoryBase<FocalPointPlasticityInternalLink> {

	public:

		FPPInternalLinkInventory() {}
		virtual ~FPPInternalLinkInventory() {}

		FocalPointPlasticityInternalLink* getLinkByCells(CellG* _cell0, CellG* _cell1) { return getLinkById(FPPLinkID(_cell0->id, _cell1->id)); }

	};
	
	class FOCALPOINTPLASTICITY_EXPORT FPPAnchorInventory : public FPPLinkInventoryBase<FocalPointPlasticityAnchor> {

	public:

		FPPAnchorInventory() {}
		virtual ~FPPAnchorInventory() {}

		FocalPointPlasticityAnchor* getAnchor(CellG* _cell, long _anchorId) { return getLinkById(FPPLinkID(_cell->id, _anchorId)); }

		int getNextAnchorId(CellG* _cell) {
			FPPLinkList ll = getCellLinkList(_cell);
			if (ll.size() == 0) return int(0);
			else return (*ll.end())->getAnchorId() + 1;
		}

	};

}

#endif