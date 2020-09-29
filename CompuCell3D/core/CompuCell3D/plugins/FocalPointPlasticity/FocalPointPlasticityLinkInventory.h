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

namespace CompuCell3D {

	/**
	Written by T.J. Sego, Ph.D.
	*/

	class FocalPointPlasticityLink;
	class FocalPointPlasticityInternalLink;
	class FocalPointPlasticityAnchor;

	template <typename T>
	class FPPLinkListBase : public std::vector<T*> {
	public:
		typedef typename std::vector<T*>::iterator FPPLinkListIterator_t;
		virtual ~FPPLinkListBase();
	};

	class FPPLinkID {
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
		virtual ~FPPLinkID();
		bool operator < (const FPPLinkID & _rhs) const { return id0 < _rhs.id0 || (id0 == _rhs.id0 && id1 < _rhs.id1); }
		bool operator == (const FPPLinkID & _rhs) const { return id0 == _rhs.id0 && id1 == _rhs.id1; }
		bool operator != (const FPPLinkID & _rhs) const { return !(operator==(_rhs)); }
	};

	template <class LinkType>
	class FPPLinkInventoryBase {
	public:

		typedef FPPLinkListBase<LinkType> FPPLinkList;

		typedef std::map<FPPLinkID, LinkType*> linkInventory_t;
		typedef typename linkInventory_t::iterator linkInventoryItr_t;
		typedef std::pair<FPPLinkID, LinkType*> linkInventoryPair_t;

		typedef std::map<CellG*, FPPLinkInventoryBase<LinkType> > cellLinkInventory_t;
		typedef typename cellLinkInventory_t::iterator cellLinkInventoryItr_t;
		typedef std::pair<CellG*, FPPLinkInventoryBase<LinkType> > cellLinkInventoryPair_t;

	protected:

		linkInventory_t linkInventory;

		LinkType* getLinkById(FPPLinkID _id) { return linkInventory[_id]; }
		FPPLinkID getLinkId(LinkType* _link) const { return FPPLinkID(getObjId0(_link), getObjId1(_link)); }

		// Get the first object attached to a link
		virtual CellG* getObj0(LinkType* _link) const;
		// Get the second object attached to a link
		virtual CellG* getObj1(LinkType* _link) const;
		// Get the ID of the first object attached to a link
		virtual long getObjId0(LinkType* _link) const;
		// Get the ID of the second object attached to a link
		virtual long getObjId1(LinkType* _link) const;

		// Add a link without any additional internal work
		void addLinkNoChain(LinkType* _link) {
			linkInventory.insert(linkInventoryPair_t(getLinkId(_link), _link));
		}
		// Remove a link without any additional internal work
		void removeLinkNoChain(LinkType* _link) {
			FPPLinkID linkId = getLinkId(_link);
			linkInventoryItr_t itr = linkInventory.find(linkId);
			if (itr != linkInventory.end()) linkInventory.erase(linkId);
		}

		cellLinkInventory_t cellLinkInventory;

		FPPLinkInventoryBase<LinkType> getCellLinkInventory(CellG* _cell) { return cellLinkInventory[_cell]; }

	public:
		FPPLinkInventoryBase() {

			linkInventory = linkInventory_t();

		};
		virtual ~FPPLinkInventoryBase() {};

		int getLinkInventorySize() { return linkInventory.size(); }
		linkInventoryItr_t linkInventoryBegin() { return linkInventory.begin(); }
		linkInventoryItr_t linkInventoryEnd() { return linkInventory.end(); }
		void incrementIterator(linkInventoryItr_t& _itr) { ++_itr; }
		void decrementIterator(linkInventoryItr_t& _itr) { --_itr; }

		// Add a link to the link inventory and update internals
		void addToInventory(LinkType* _link) {
			addLinkNoChain(_link);
			getCellLinkInventory(getObj0(_link)).addLinkNoChain(_link);
			getCellLinkInventory(getObj1(_link)).addLinkNoChain(_link);
		}
		// Remove a link to the link inventory and update internals
		void removeFromInventory(LinkType* _link) {
			removeLinkNoChain(_link);
			getCellLinkInventory(getObj0(_link)).removeLinkNoChain(_link);
			getCellLinkInventory(getObj1(_link)).removeLinkNoChain(_link);
		}

		// Get link inventory list
		FPPLinkList getLinkList() { 
			FPPLinkList fppLinkList;
			for (linkInventoryItr_t itr = linkInventory.begin(); itr != linkInventory.end(); ++itr)
				fppLinkList.push_back(itr->second);
			return fppLinkList;
		}
		// Get link inventory list by cell
		FPPLinkList getCellLinkList(CellG* _cell) { return getCellLinkInventory(_cell).getLinkList(); }
		// Remove all links attached to a cell
		void removeCellLinks(CellG* _cell) {
			FPPLinkList cellLinkList = getCellLinkList(_cell);
			for (FPPLinkList::iterator itr = cellLinkList.begin(); itr != cellLinkList.end(); ++itr)
				removeFromInventory((*itr));
			
			cellLinkInventoryItr_t cItr = cellLinkInventory.find(_cell);
			if (cItr != cellLinkInventory.end()) cellLinkInventory.erase(_cell);
		}

	};

	class FPPLinkInventory : public FPPLinkInventoryBase<FocalPointPlasticityLink> {

	protected:

		CellG* getObj0(FocalPointPlasticityLink* _link) const { return _link->initiator; }
		CellG* getObj1(FocalPointPlasticityLink* _link) const { return _link->initiated; }
		long getObjId0(FocalPointPlasticityLink* _link) const { return _link->getId0(); }
		long getObjId1(FocalPointPlasticityLink* _link) const { return _link->getId1(); }

	public:

		FPPLinkInventory() {}
		virtual ~FPPLinkInventory() {}

		FocalPointPlasticityLink* getLinkByCells(CellG* _cell0, CellG* _cell1) {
			FPPLinkList linkList = getCellLinkList(_cell0);
			FPPLinkID linkId(_cell0->id, _cell1->id);
			for (FPPLinkList::iterator itr = linkList.begin(); itr != linkList.end(); ++itr)
				if (getLinkId((*itr)) == linkId) return (*itr);

			return (FocalPointPlasticityLink*)(0);
		}

	};

	class FPPInternalLinkInventory : public FPPLinkInventoryBase<FocalPointPlasticityInternalLink> {

	protected:

		CellG* getObj0(FocalPointPlasticityInternalLink* _link) const { return _link->initiator; }
		CellG* getObj1(FocalPointPlasticityInternalLink* _link) const { return _link->initiated; }
		long getObjId0(FocalPointPlasticityInternalLink* _link) const { return _link->getId0(); }
		long getObjId1(FocalPointPlasticityInternalLink* _link) const { return _link->getId1(); }

	public:

		FPPInternalLinkInventory() {}
		virtual ~FPPInternalLinkInventory() {}

		FocalPointPlasticityInternalLink* getLinkByCells(CellG* _cell0, CellG* _cell1) {
			FPPLinkList linkList = getCellLinkList(_cell0);
			FPPLinkID linkId(_cell0->id, _cell1->id);
			for (FPPLinkList::iterator itr = linkList.begin(); itr != linkList.end(); ++itr)
				if (getLinkId((*itr)) == linkId) return (*itr);

			return (FocalPointPlasticityInternalLink*)(0);
		}

	};

	class FPPAnchorInventory : public FPPLinkInventoryBase<FocalPointPlasticityAnchor> {

		CellG *mediumPointer;

	protected:

		CellG* getObj0(FocalPointPlasticityAnchor* _link) const { return _link->initiator; }
		CellG* getObj1(FocalPointPlasticityAnchor* _link) const { return mediumPointer; }
		long getObjId0(FocalPointPlasticityAnchor* _link) const { return _link->getId0(); }
		long getObjId1(FocalPointPlasticityAnchor* _link) const { return _link->getId1(); }

	public:

		FPPAnchorInventory() :
			mediumPointer((CellG*)(0))
		{}
		virtual ~FPPAnchorInventory() {}

	};

}

#endif