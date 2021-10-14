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
#ifndef FOCALPOINTPLASTICITYLINKINVENTORYBASE_H
#define FOCALPOINTPLASTICITYLINKINVENTORYBASE_H

#include <unordered_map>
#include <set>
#include <vector>

#include <CompuCell3D/CC3D.h>

#include "FocalPointPlasticityLinks.h"
#include "FocalPointPlasticityDLLSpecifier.h"

namespace CompuCell3D {

	/**
	Written by T.J. Sego, Ph.D.
	*/

	class Potts3D;
	class FocalPointPlasticityTracker;

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

		FPPLinkID() : id0(0), id1(0) {};
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
		~FPPLinkID() {}
		bool operator < (const FPPLinkID & _rhs) const { return id0 < _rhs.id0 || (id0 == _rhs.id0 && id1 < _rhs.id1); }
		bool operator == (const FPPLinkID & _rhs) const { return id0 == _rhs.id0 && id1 == _rhs.id1; }
		bool operator != (const FPPLinkID & _rhs) const { return !(operator==(_rhs)); }
	};

	template <class LinkType> class FPPLinkInventoryBase;

	template <class LinkType>
	class FOCALPOINTPLASTICITY_EXPORT FPPLinkInventoryTracker {
	public:

		FPPLinkInventoryTracker() {};
		~FPPLinkInventoryTracker() {};
		
		FPPLinkInventoryBase<LinkType> linkInv;
	};

	
	// Hasher using Cantor pairing function
	class FOCALPOINTPLASTICITY_EXPORT LinkInventoryHasher {

	public:

		size_t operator() (FPPLinkID key) const {
			return 0.5 * (key.id0 + key.id1) * (key.id0 + key.id1 + 1) + key.id1;
		}

	};


	template <class LinkType>
	class FOCALPOINTPLASTICITY_EXPORT FPPLinkInventoryBase {

	public:

		typedef FPPLinkListBase<LinkType> FPPLinkList;
		typedef FPPLinkInventoryBase<LinkType> FPPInventory_t;

		typedef std::unordered_map<const FPPLinkID, LinkType*, LinkInventoryHasher> linkInventory_t;
		typedef typename linkInventory_t::iterator linkInventoryItr_t;
		typedef std::pair<const FPPLinkID, LinkType*> linkInventoryPair_t;

	protected:

		Potts3D* potts;
		ExtraMembersGroupAccessor<FPPLinkInventoryTracker<LinkType> >* cellLinkInventoryTrackerAccessor;

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

	public:
		FPPLinkInventoryBase() {}
		FPPLinkInventoryBase(ExtraMembersGroupAccessor<FPPLinkInventoryTracker<LinkType> >* _cellLinkInventoryTrackerAccessor, Potts3D* _potts)
		{
			cellLinkInventoryTrackerAccessor = _cellLinkInventoryTrackerAccessor;
			potts = _potts;
		}
		virtual ~FPPLinkInventoryBase() {}

		ExtraMembersGroupAccessor<FPPLinkInventoryTracker<LinkType> >* getFocalPointPlasticityCellLinkInventoryTrackerAccessorPtr() { return cellLinkInventoryTrackerAccessor; }

		virtual std::set<FocalPointPlasticityTrackerData> getFPPTrackerDataSet(CellG* _cell)
		{
			std::set<FocalPointPlasticityTrackerData> o;
            for (auto & link : getCellLinkList(_cell)) {
                o.insert(link->getFPPTrackerData(_cell));
            }

			return o;
		}

		virtual int getLinkInventorySize() { return linkInventory.size(); }
		linkInventory_t& getContainer() { return linkInventory; }
		virtual linkInventoryItr_t linkInventoryBegin() { return linkInventory.begin(); }
		virtual linkInventoryItr_t linkInventoryEnd() { return linkInventory.end(); }

		// Add a link to the link inventory and update internals
		void addToInventory(LinkType* _link) {
			addLinkNoChain(_link);
			CellG* linkObj0 = _link->getObj0();
			CellG* linkObj1 = _link->getObj1();
			if (linkObj0) getCellLinkInventory(linkObj0)->addLinkNoChain(_link);
			if (linkObj1) getCellLinkInventory(linkObj1)->addLinkNoChain(_link);
		}
		// Remove a link to the link inventory and update internals
		void removeFromInventory(LinkType* _link) {
			removeLinkNoChain(_link);
			CellG* linkObj0 = _link->getObj0();
			CellG* linkObj1 = _link->getObj1();
			if (linkObj0) getCellLinkInventory(linkObj0)->removeLinkNoChain(_link);
			if (linkObj1) getCellLinkInventory(linkObj1)->removeLinkNoChain(_link);
			delete _link;
		}

		// Get link inventory list
		FPPLinkList getLinkList() {
			FPPLinkList fppLinkList;
			for (linkInventoryItr_t itr = linkInventory.begin(); itr != linkInventory.end(); ++itr)
				fppLinkList.push_back(itr->second);
			return fppLinkList;
		}
		FPPInventory_t* getCellLinkInventory(CellG* _cell) { 
			return &cellLinkInventoryTrackerAccessor->get(_cell->extraAttribPtr)->linkInv;
		}
		// Get link inventory list by cell
		FPPLinkList getCellLinkList(CellG* _cell) { return getCellLinkInventory(_cell)->getLinkList(); }
		// Get number of junctions for a cell
		virtual int getNumberOfJunctions(CellG* _cell) { return getCellLinkInventory(_cell)->getLinkInventorySize(); }
		// Remove all links attached to a cell
		void removeCellLinks(CellG* _cell) {
			FPPLinkList cellLinkList = getCellLinkList(_cell);
//			for (FPPLinkList::iterator itr = cellLinkList.begin(); itr != cellLinkList.end(); ++itr)
//				removeFromInventory((*itr));

            for (auto & link : cellLinkList) {
                removeFromInventory(link);
            }


		}

	};

}

#endif